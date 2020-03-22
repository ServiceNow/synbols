import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
from .modules.protonet import prototype_distance
from .backbones import get_backbone
import higher


class MAML(torch.nn.Module):
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
        
        ## for now, one task at a time
        task_num = 1
        n_inner_iter = 5
        
        qry_losses = []
        qry_accs = []
        
        self.backbone.train()
        for episode in tqdm(loader):
            episode = episode[0] # undo collate
            self.optimizer.zero_grad()
            support_set = episode["support_set"].cuda(non_blocking=False)
            query_set = episode["query_set"].cuda(non_blocking=False)

            ss, nclasses, c, h, w = support_set.size()
            qs, nclasses, c, h, w = query_set.size()
            
            absolute_labels = episode["targets"]
            relative_labels = absolute_labels.clone()
            # TODO: use episode['targets']
            support_relative_labels = torch.arange(episode['nclasses']).view(1, -1).repeat(
                episode['support_size'], 1).cuda().view(-1)
            query_relative_labels = torch.arange(episode['nclasses']).view(1, -1).repeat(
                episode['query_size'], 1).cuda().view(-1)
            
            inner_opt = torch.optim.SGD(self.backbone.parameters(), lr=1e-1)
            
            querysz = query_relative_labels.size()[0]

            self.optimizer.zero_grad()
            for i in range(task_num):
                with higher.innerloop_ctx(self.backbone, inner_opt,
                    copy_initial_weights=False) as (fnet, diffopt):
                    
                    for _ in range(n_inner_iter):
                        ## for now only one task at a time
                        spt_logits = fnet(support_set.view(ss * nclasses, c, h, w)).view(ss * nclasses, -1)
                        spt_loss = F.cross_entropy(spt_logits, support_relative_labels)
                        diffopt.step(spt_loss)

                    qry_logits = fnet(query_set.view(qs * nclasses, c, h, w))
                    qry_loss = F.cross_entropy(qry_logits, query_relative_labels)
                    qry_losses.append(qry_loss.detach())
                    qry_acc = (qry_logits.argmax(
                        dim=1) == query_relative_labels).sum().item() / querysz
                    qry_accs.append(qry_acc)

                    qry_loss.backward()

            self.optimizer.step()

        qry_losses = sum(qry_losses) / len(qry_losses)
        qry_accs = 100. * sum(qry_accs) / len(qry_accs)

        return {"train_loss": qry_losses.item(), "train_acc":qry_accs}

    #@torch.no_grad()
    def val_on_loader(self, loader, savedir=None):
        

        ## for now, one task at a time
        task_num = 1
        n_inner_iter = 5

        qry_losses = []
        qry_accs = []
        
        # TDOO: if I put the model in eval(), the model explods...
        #self.backbone.eval()
        self.backbone.train()
        for episode in tqdm(loader):
            episode = episode[0] # undo collate
            support_set = episode["support_set"].cuda(non_blocking=False)
            query_set = episode["query_set"].cuda(non_blocking=False)

            ss, nclasses, c, h, w = support_set.size()
            qs, nclasses, c, h, w = query_set.size()
            
            if ss != episode["support_size"] or qs != episode["query_size"]:
                raise(RuntimeError("The dataset is too small for the current support and query sizes"))
            
            absolute_labels = episode["targets"]
            relative_labels = absolute_labels.clone()
            # TODO: use episode['targets']
            support_relative_labels = torch.arange(episode['nclasses']).view(1, -1).repeat(
                episode['support_size'], 1).cuda().view(-1)
            query_relative_labels = torch.arange(episode['nclasses']).view(1, -1).repeat(
                episode['query_size'], 1).cuda().view(-1)

            inner_opt = torch.optim.SGD(self.backbone.parameters(), lr=1e-1)
            
            querysz = query_relative_labels.size()[0]

            for i in range(task_num):
                with higher.innerloop_ctx(self.backbone, inner_opt,
                    track_higher_grads=False) as (fnet, diffopt):
                    
                    for _ in range(n_inner_iter):
                        ## for now only one task at a time
                        spt_logits = fnet(support_set.view(ss * nclasses, c, h, w)).view(ss * nclasses, -1)
                        spt_loss = F.cross_entropy(spt_logits, support_relative_labels)
                        # print(spt_logits)
                        # print(spt_loss)
                        diffopt.step(spt_loss)

                    qry_logits = fnet(query_set.view(qs * nclasses, c, h, w))
                    qry_loss = F.cross_entropy(qry_logits, query_relative_labels)
                    qry_losses.append(qry_loss.detach())
                    qry_acc = (qry_logits.argmax(
                        dim=1) == query_relative_labels).sum().item() / querysz
                    qry_accs.append(qry_acc)
        
        qry_losses = sum(qry_losses) / len(qry_losses)
        qry_accs = 100. * sum(qry_accs) / len(qry_accs)

        self.scheduler.step(qry_losses)
        return {"val_loss": qry_losses.item(), 
                "val_accuracy": qry_accs}

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
