import torch
import torchvision.models as models

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
from backbones import get_backbone
import time


def get_model(exp_dict):
    if exp_dict["model"] == 'classification':
        return Classification(exp_dict) 
    else:
        raise ValueError("Model %s not found" %exp_dict["model"])

class Classification(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.exp_dict = exp_dict
        self.backbone = get_backbone(exp_dict)
        self.backbone.cuda()
        
        self.optimizer = torch.optim.Adam(self.backbone.parameters(),
                                            lr=exp_dict['lr'],
                                            weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=10,
                                                                    verbose=True)

    def train_on_loader(self, loader):
        _loss = 0
        _total = 0
        _batch_time = []
        self.backbone.train()
        for x, y in tqdm(loader):
            t = time.time()
            self.optimizer.zero_grad()
            y = y.cuda(non_blocking=True)
            x = x.cuda(non_blocking=False)
            if self.exp_dict["backbone"]["name"] == "warn":
                logits, regularizer = self.backbone(x)
            else:
                logits = self.backbone(x)
                regularizer = 0
            loss = F.cross_entropy(logits, y) + regularizer
            _batch_time.append(time.time() - t)
            _loss += float(loss) * x.size(0)
            _total += x.size(0)
            loss.backward()
            self.optimizer.step()
        return {"train_loss": float(_loss) / _total,
                "train_epoch_time": sum(_batch_time),
                "train_batch_time": np.mean(_batch_time)}

    @torch.no_grad()
    def val_on_loader(self, loader, savedir=None):
        self.backbone.eval()
        _accuracy = 0
        _total = 0
        _loss = 0
        _batch_time = []
        for x, y in tqdm(loader):
            t = time.time()
            y = y.cuda(non_blocking=True)
            x = x.cuda(non_blocking=False)
            if self.exp_dict["backbone"]["name"] == "warn":
                logits, regularizer = self.backbone(x)
            else:
                logits = self.backbone(x)
                regularizer = 0
            preds = logits.data.max(-1)[1]
            loss = F.cross_entropy(logits, y)
            _batch_time.append(time.time() - t)
            _loss += float(loss) * x.size(0)
            _accuracy += float((preds == y).float().sum())
            _total += x.size(0)
        _loss /= _total
        _accuracy /= _total
        self.scheduler.step(_loss)
        return {"val_loss": _loss,
                "val_accuracy": _accuracy,
                "val_epoch_time": sum(_batch_time),
                "val_batch_time": np.mean(_batch_time)}

    def get_state_dict(self):
        state = {}
        state["model"] = self.backbone.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        state["scheduler"] = self.scheduler.state_dict()
        return state

    def set_state_dict(self, state_dict):
        self.backbone.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])