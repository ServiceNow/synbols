import torch
import torchvision.models as models

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
from backbones import get_backbone
import time
try:
    from apex import amp
except:
    pass

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
        self.min_lr = exp_dict["lr"] * exp_dict["min_lr_decay"]
        self.optimizer = torch.optim.Adam(self.backbone.parameters(),
                                            lr=exp_dict['lr'],
                                            weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=10,
                                                                    min_lr=self.min_lr,
                                                                    verbose=True)
        if self.exp_dict["amp"] > 0:
            self.backbone, self.optimizer = amp.initialize(self.backbone, self.optimizer, opt_level="O%d" %self.exp_dict["amp"])

    def is_end(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
        return lr <= self.min_lr


    def backward(self, loss):
        if self.exp_dict["amp"] > 0:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
    def vis_on_batch(self, x, y, name, gridsize=5):
        import pylab
        y = y.data.cpu().numpy()
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).data.cpu().numpy()
        x -= x.min()
        x /= x.max()
        y_set = np.unique(y)
        rows = len(y_set)
        cols = int(np.max([(y == _y).astype(float).sum() for _y in y_set]))
        ret = np.zeros((rows, cols, h, w, 3))
        for row, _y in enumerate(y_set):
            _x = x[y == _y]
            ret[row, :_x.shape[0], ...] = _x
        pylab.imsave(name, ret.transpose((0, 2, 1, 3, 4)).reshape((h*rows, w*cols, 3)))

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
            self.backward(loss)
            self.optimizer.step()
        # self.vis_on_batch(x, y, 'train.png')
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
        # self.vis_on_batch(x, y, 'val.png')
        return {"val_loss": _loss,
                "val_accuracy": _accuracy,
                "val_epoch_time": sum(_batch_time),
                "val_batch_time": np.mean(_batch_time)}

    def get_state_dict(self):
        state = {}
        state["model"] = self.backbone.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        state["scheduler"] = self.scheduler.state_dict()
        if self.exp_dict["amp"] > 0:
            state["amp"] = amp.state_dict()
        return state

    def set_state_dict(self, state_dict):
        self.backbone.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        if self.exp_dict["amp"] > 0:
            amp.load_state_dict(state_dict["amp"])