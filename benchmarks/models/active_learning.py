import os
import types
from copy import deepcopy

import h5py
import numpy as np
import torch
import torch.utils.data as torchdata
from baal import ModelWrapper
from baal.active import ActiveLearningLoop, get_heuristic
from baal.calibration import DirichletCalibrator
from baal.utils.metrics import ClassificationReport
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import models

from datasets import get_dataset

pjoin = os.path.join

LOG_UNCERT = False


class MyLoop(ActiveLearningLoop):
    def step(self) -> bool:
        pool = self.dataset.pool
        assert self.max_sample == -1
        if len(pool) > 0:

            # Limit number of samples
            if self.max_sample != -1 and self.max_sample < len(pool):
                indices = np.random.choice(len(pool), self.max_sample, replace=False)
                pool = torchdata.Subset(pool, indices)
            else:
                indices = np.arange(len(pool))

            probs = self.get_probabilities(pool, **self.kwargs)
            if probs is not None:
                if isinstance(probs, types.GeneratorType):
                    self.uncertainty = self.heuristic.get_uncertainties_generator(probs)
                elif len(probs) > 0:
                    self.uncertainty = self.heuristic.get_uncertainties(probs)
                else:
                    return False
                to_label = self.heuristic.reorder_indices(self.uncertainty)
                to_label = indices[np.array(to_label)]
                if len(to_label) > 0:
                    self.dataset.label(to_label[: self.ndata_to_label])
                    return True
        return False


class ActiveLearning(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.backbone = models.vgg16(pretrained=exp_dict["imagenet_pretraining"], progress=True)
        num_ftrs = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = torch.nn.Linear(num_ftrs, exp_dict["num_classes"])
        self.initial_weights = deepcopy(self.backbone.state_dict())
        self.backbone.cuda()

        self.batch_size = exp_dict['batch_size']
        self.calibrate = exp_dict.get('calibrate', False)
        self.learning_epoch = exp_dict['learning_epoch']
        self.optimizer = torch.optim.SGD(self.backbone.parameters(),
                                         lr=exp_dict['lr'],
                                         weight_decay=5e-4,
                                         momentum=0.9,
                                         nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=10,
                                                                    verbose=True)
        self.criterion = CrossEntropyLoss()
        self.heuristic = get_heuristic(exp_dict['heuristic'])
        self.wrapper = ModelWrapper(self.backbone, criterion=self.criterion)
        self.wrapper.add_metric('cls_report', lambda: ClassificationReport(exp_dict["num_classes"]))
        self.loop = MyLoop(None, self.wrapper.predict_on_dataset_generator,
                           heuristic=self.heuristic,
                           ndata_to_label=exp_dict['query_size'],
                           batch_size=1,
                           iterations=exp_dict['iterations'],
                           use_cuda=True)

        self.calib_set = get_dataset('calib', exp_dict['dataset'])
        self.valid_set = get_dataset('val', exp_dict['dataset'])
        self.calibrator = DirichletCalibrator(self.wrapper, exp_dict["num_classes"],
                                              lr=0.001, reg_factor=exp_dict['reg_factor'],
                                              mu=exp_dict['mu'])

        self.active_dataset = None
        self.active_dataset_settings = None

    def train_on_loader(self, loader: DataLoader):
        self.wrapper.load_state_dict(self.initial_weights)
        if self.active_dataset is None:
            self.active_dataset = loader.dataset
            if self.active_dataset_settings is not None:
                self.active_dataset.load_state_dict(self.active_dataset_settings)
            self.loop.dataset = self.active_dataset

        self.wrapper.train_on_dataset(self.active_dataset, self.optimizer, self.batch_size,
                                      epoch=self.learning_epoch, use_cuda=True)

        metrics = self.wrapper.metrics
        return self._format_metrics(metrics, 'train')

    def val_on_loader(self, loader, savedir=None):
        val_data = loader.dataset
        self.loop.step()
        self.wrapper.test_on_dataset(val_data, batch_size=self.batch_size, use_cuda=True)
        metrics = self.wrapper.metrics
        mets = self._format_metrics(metrics, 'test')
        mets.update({'num_samples': len(self.active_dataset)})
        return mets

    def on_train_end(self, savedir, epoch):
        h5_path = pjoin(savedir, 'ckpt.h5')
        labelled = self.active_dataset.state_dict()['labelled']
        with h5py.File(h5_path, 'a') as f:
            if f'epoch_{epoch}' not in f:
                g = f.create_group(f'epoch_{epoch}')
                g.create_dataset('labelled', data=labelled.astype(np.bool))

    def _format_metrics(self, metrics, step):
        mets = {k: v.value for k, v in metrics.items() if step in k}
        mets_unpacked = {}
        for k, v in mets.items():
            if isinstance(v, float):
                mets_unpacked[k] = v
            elif isinstance(v, np.ndarray):
                mets_unpacked[k] = v.mean()
            else:
                mets_unpacked.update({f"{k}_{ki}": np.mean(vi) for ki, vi in v.items()})
        return mets_unpacked

    def get_state_dict(self):
        state = {}
        state["model"] = self.backbone.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        state["scheduler"] = self.scheduler.state_dict()
        if self.active_dataset is None:
            state['dataset'] = None
        else:
            state["dataset"] = self.active_dataset.state_dict()
        return state

    def set_state_dict(self, state_dict):
        self.backbone.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.active_dataset_settings = state_dict["dataset"]
        if self.active_dataset is not None:
            self.active_dataset.load_state_dict(self.active_dataset_settings)
