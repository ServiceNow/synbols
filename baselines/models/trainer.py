import torch
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm


class Trainer(torch.nn.Module):
    def __init__(self, num_classes, exp_dict):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False, 
                                        num_classes=num_classes).cuda()
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
        for x, y in tqdm(loader):
            self.optimizer.zero_grad()
            y = y.cuda(non_blocking=True)
            x = x.cuda(non_blocking=False)
            logits = self.backbone(x)
            loss = F.cross_entropy(logits, y)
            _loss += float(loss)
            _total += x.size(0)
            loss.backward()
            self.optimizer.step()
        return {"train_loss": float(_loss) / _total}

    @torch.no_grad()
    def val_on_loader(self, loader, savedir=None):
        _accuracy = 0
        _total = 0
        _loss = 0
        for x, y in tqdm(loader):
            y = y.cuda(non_blocking=True)
            x = x.cuda(non_blocking=False)
            logits = self.backbone(x)
            loss = F.cross_entropy(logits, y)
            _loss += float(loss)
            _accuracy += float((logits.data.max(-1)[1] == y).float().sum())
            _total += x.size(0)
        self.scheduler.step(_loss / _total)
        return {"val_loss": _loss / _total, 
                "val_accuracy": _accuracy / _total}

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
