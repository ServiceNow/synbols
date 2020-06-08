import os
from copy import deepcopy

import h5py
import numpy as np
import torch
from baal import ModelWrapper
from baal.active import ActiveLearningLoop, get_heuristic
from baal.bayesian.dropout import patch_module
from baal.calibration import DirichletCalibrator
from baal.utils.metrics import ClassificationReport, Metrics
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import models

from datasets import get_dataset

pjoin = os.path.join

LOG_UNCERT = False


class Accuracy(Metrics):
    def __init__(self):
        super().__init__(average=False)
        self._tp = 0
        self._count = 0

    def reset(self):
        self._tp = 0
        self._count = 0

    def update(self, output=None, target=None):
        """
        Update TP and support.

        Args:
            output (tensor): predictions of model
            target (tensor): labels

        Raises:
            ValueError if the first dimension of output and target don't match.
        """
        batch_size = target.shape[0]
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy().reshape(-1)
        output = np.argmax(output, axis=1).reshape(-1)
        self._count += batch_size
        self._tp += (target == output).sum()

    @property
    def value(self):
        return self._tp / self._count


class ActiveLearning(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.backbone = models.vgg16(pretrained=exp_dict["imagenet_pretraining"], progress=True)
        num_ftrs = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = torch.nn.Linear(num_ftrs, exp_dict["num_classes"])
        self.backbone = patch_module(self.backbone)
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

        self.criterion = CrossEntropyLoss()
        shuffle_prop = exp_dict.get('shuffle_prop', 0.0)
        max_sample = -1
        self.heuristic = get_heuristic(exp_dict['heuristic'], shuffle_prop=shuffle_prop)
        self.wrapper = ModelWrapper(self.backbone, criterion=self.criterion)
        self.wrapper.add_metric('cls_report', lambda: ClassificationReport(exp_dict["num_classes"]))
        self.wrapper.add_metric('accuracy', lambda: Accuracy())
        self.loop = ActiveLearningLoop(None, self.wrapper.predict_on_dataset,
                                       heuristic=self.heuristic,
                                       ndata_to_label=exp_dict['query_size'],
                                       batch_size=self.batch_size,
                                       iterations=exp_dict['iterations'],
                                       use_cuda=True,
                                       max_sample=max_sample)

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
        self.criterion.train()
        self.wrapper.train_on_dataset(self.active_dataset, self.optimizer, self.batch_size,
                                      epoch=self.learning_epoch, use_cuda=True)

        metrics = self.wrapper.metrics
        return self._format_metrics(metrics, 'train')

    def val_on_loader(self, loader, savedir=None):
        val_data = loader.dataset
        self.loop.step()
        self.criterion.eval()
        self.wrapper.test_on_dataset(val_data, batch_size=self.batch_size, use_cuda=True,
                                     average_predictions=20)
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


class CalibratedActiveLearning(ActiveLearning):
    def train_on_loader(self, loader: DataLoader):
        self.wrapper.load_state_dict(self.initial_weights)
        if self.active_dataset is None:
            self.active_dataset = loader.dataset
            if self.active_dataset_settings is not None:
                self.active_dataset.load_state_dict(self.active_dataset_settings)
            self.loop.dataset = self.active_dataset

        self.wrapper.train_on_dataset(self.active_dataset,
                                      self.optimizer,
                                      self.batch_size,
                                      epoch=self.learning_epoch,
                                      use_cuda=True)

        metrics = self.wrapper.metrics
        return self._format_metrics(metrics, 'train')

    def val_on_loader(self, loader, savedir=None):
        val_data = loader.dataset

        if self.calibrate:
            self.calibrator.calibrate(self.calib_set, self.valid_set,
                                      batch_size=16, epoch=10, use_cuda=True,
                                      double_fit=True)
            calibrated_model = ModelWrapper(self.calibrator.calibrated_model,
                                            None)
            self.loop.get_probabilities = calibrated_model.predict_on_dataset
        self.loop.step()
        self.wrapper.test_on_dataset(val_data, batch_size=self.batch_size,
                                     use_cuda=True, average_predictions=20)
        metrics = self.wrapper.metrics
        mets = self._format_metrics(metrics, 'test')
        mets.update({'num_samples': len(self.active_dataset)})
        return mets
