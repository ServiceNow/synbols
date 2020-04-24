import torch
import numpy as np
import torch.nn.functional as F

class make_RelationNet(torch.nn.Module):
    def __init__(self, exp_dict, F, G):
        super().__init__()

        self._F = F
        self._G = G

    # def forward(self, x_left, x_right, nclasses, nqueries):
    def forward(self, support_set, query_set, labels):
        ## for the Tensor gymnastic, see:
        ## https://github.com/dragen1860/LearningToCompare-Pytorch/blob/master/compare.py

        support_emb = self._F(support_set)
        query_emb = self._F(query_set)

        supportsz, hiddensz = support_emb.size()
        querysz, _ = query_emb.size()

        ## concat each query with all support along dim = latent representation
        support_emb = support_emb.unsqueeze(0).expand(querysz, -1, -1)
        query_emb = query_emb.unsqueeze(1).expand(-1, supportsz, -1)
        comb = torch.cat([support_emb, query_emb], dim=2)

        ## compute score
        score = self._G(comb).squeeze(2) 

        return score 

def RelationNet_acc(support_y, query_y, score):

    score = score.detach()

    _, predict_idx = torch.max(score,1)
    predict_labels = support_y[predict_idx]
    acc = torch.eq(predict_labels,query_y).float().mean()    
    
    return acc
