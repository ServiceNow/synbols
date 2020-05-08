import torch
import torch.nn.functional as F
# from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
from .modules.RelationNet import make_RelationNet, RelationNet_acc 
from .backbones import get_backbone
import os

class RelationNet(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        
        _F = get_backbone(exp_dict, 
                            architecture="conv4",
                            hidden_size=exp_dict["hidden_size"],
                            feature_extractor=True)
        _G = get_backbone(exp_dict,
                            architecture="mlp3",
                            hidden_size=2*exp_dict["hidden_size"],
                            output_size=1,
                            feature_extractor=False)
        
        self.backbone = make_RelationNet(exp_dict, _F, _G)
        
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
        _accuracy = 0
        _total = 0

        # self.temp += 1
        self.backbone.train()
        for episode in tqdm(loader):

            ## Boilerplate
            episode = episode[0] # undo collate
            # plot_episode(episode, classes_first=False, epoch=self.temp)
            support_set = episode["support_set"].cuda(non_blocking=False)
            query_set = episode["query_set"].cuda(non_blocking=False)

            ss, nclasses, c, h, w = support_set.size()
            qs, nclasses, c, h, w = query_set.size()

            absolute_labels = episode["targets"]
            relative_labels = absolute_labels.clone()
            
            # TODO: use episode['targets']
            support_rel_labels = torch.arange(episode['nclasses']).view(1, -1).repeat(
                episode['support_size'], 1).cuda().view(-1)
            query_rel_labels = torch.arange(episode['nclasses']).view(1, -1).repeat(
                episode['query_size'], 1).cuda().view(-1)
            
            ## Forward Pass
            self.optimizer.zero_grad()
            support_set = support_set.view(ss * nclasses, c, h, w)
            query_set = query_set.view(qs * nclasses, c, h, w)
            score = self.backbone(support_set, query_set, support_rel_labels)

            # print(score)

            ## create labels
            support_y = support_rel_labels.unsqueeze(0).expand(qs*nclasses, -1) 
            query_y = query_rel_labels.unsqueeze(1).expand(-1, ss*nclasses)
            labels = torch.eq(support_y, query_y).float() 

            ## loss and bprop
            loss = ((labels - score)**2).mean()
            loss.backward()
            self.optimizer.step()

            accuracy = RelationNet_acc(support_rel_labels, query_rel_labels, score)
            _accuracy += float(accuracy)
            _loss += float(loss)
            _total += 1

        return {"train_loss": float(_loss) / _total,
                "train_accuracy": 100*(float(_accuracy) / _total)}

    @torch.no_grad()
    def val_on_loader(self, loader, savedir=None):
        _accuracy = 0
        _total = 0
        _loss = 0
        _logits = []
        _targets = []
        
        self.backbone.eval()
        for episode in tqdm(loader):
            
            ## Boilerplate
            episode = episode[0] # undo collate
            support_set = episode["support_set"].cuda(non_blocking=False)
            query_set = episode["query_set"].cuda(non_blocking=False)

            ss, nclasses, c, h, w = support_set.size()
            qs, nclasses, c, h, w = query_set.size()

            if ss != episode["support_size"] or qs != episode["query_size"]:
                raise(RuntimeError(
                    "The dataset is too small for the current support and query sizes"))

            absolute_labels = episode["targets"]
            relative_labels = absolute_labels.clone()
            
            # TODO: use episode['targets']
            support_rel_labels = torch.arange(episode['nclasses']).view(1, -1).repeat(
                episode['support_size'], 1).cuda().view(-1)
            query_rel_labels = torch.arange(episode['nclasses']).view(1, -1).repeat(
                episode['query_size'], 1).cuda().view(-1)

            ## Forward pass
            support_set = support_set.view(ss * nclasses, c, h, w)
            query_set = query_set.view(qs * nclasses, c, h, w)
            score = self.backbone(support_set, query_set, support_rel_labels)
            
            ## create labels
            support_y = support_rel_labels.unsqueeze(0).expand(qs*nclasses, -1) 
            query_y = query_rel_labels.unsqueeze(1).expand(-1, ss*nclasses)
            labels = torch.eq(support_y, query_y).float() 

            ## loss and eval 
            loss = ((labels - score)**2).mean()
            accuracy = RelationNet_acc(support_rel_labels, query_rel_labels, score)
            _loss += float(loss) * qs * nclasses
            _accuracy += float(accuracy) * qs * nclasses
            _total += qs * nclasses
        
        self.scheduler.step(_loss / _total)
        
        return {"val_loss": _loss / _total,
                "val_accuracy": 100*(_accuracy / _total)}

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



