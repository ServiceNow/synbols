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
        self.backbone = get_backbone(exp_dict)
        self.backbone.cuda()
        
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

    def train_on_loader(self, loader):
        _loss = 0
        _total = 0
        self.backbone.train()
        t = time.time()
        for x, y in tqdm(loader):
            self.optimizer.zero_grad()
            y = y.cuda(non_blocking=True)
            x = x.cuda(non_blocking=False)
            logits = self.backbone(x)
            loss = F.cross_entropy(logits, y)
            _loss += float(loss) * x.size(0)
            _total += x.size(0)
            loss.backward()
            self.optimizer.step()
        time_epoch = time.time() - t
        return {"train_loss": float(_loss) / _total,
                "train_epoch_time": time_epoch}

    @torch.no_grad()
    def val_on_loader(self, loader, savedir=None):
        self.backbone.eval()
        _accuracy = 0
        _total = 0
        _loss = 0
        t = time.time()
        for x, y in tqdm(loader):
            y = y.cuda(non_blocking=True)
            x = x.cuda(non_blocking=False)
            logits = self.backbone(x)
            preds = logits.data.max(-1)[1]
            loss = F.cross_entropy(logits, y)
            _loss += float(loss) * x.size(0)
            _accuracy += float((preds == y).float().sum())
            _total += x.size(0)
        epoch_time = time.time() - t
        self.scheduler.step(_loss / _total)
        return {"val_loss": _loss / _total,
                "val_accuracy": _accuracy / _total,
                "val_epoch_time": epoch_time}

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