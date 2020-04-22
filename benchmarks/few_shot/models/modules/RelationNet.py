import torch
import numpy as np
import torch.nn.functional as F

# class RelationNet(Summary):
class RelationNet(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
    # def __init__(self, ni, no, depth1, depth2, depth3, norm=torch.nn.BatchNorm2d, activation=F.relu, **kwargs):
        # super().__init__(ni, no)
        self.depth1 = int(exp_dict['depth1'])
        self.depth2 = int(exp_dict['depth2'])
        self.depth3 = int(exp_dict['depth3'])

        self.ni = exp_dict['ni']
        self.no = exp_dict['no']

        self.activation = F.relu 
        self.norm = torch.nn.BatchNorm2d 

        class _G(torch.nn.Module):
            def __init__(self, ni, no, depth):
                super().__init__()
                self.depth = depth
                self.ni = ni
                self.no = no
                for i in range(self.depth):
                    setattr(self, "bn_%d" % i, nn.BatchNorm1d(self.ni))
                    setattr(self, "layer_%d" % i, nn.Linear(self.ni, self.no))
                    self.ni = self.no
            def forward(self, x):
                for i in range(self.depth):
                    bn = getattr(self, "bn_%d" % i)
                    layer = getattr(self, "layer_%d" % i)
                    x = x + layer(F.relu(bn(x), True))
                return x
        
        class _F(torch.nn.Module):
            def __init__(self, ni, no, depth):
                super().__init__()
                self.depth = depth
                self.ni = ni
                self.no = no
                for i in range(self.depth):
                    setattr(self, "layer_%d" % i, torch.nn.Linear(self.ni, self.no))
                    self.ni = self.no
            def forward(self, x):
                if self.depth == 0:
                    return x
                x = self.layer_0(F.relu(x, True))
                for i in range(1, self.depth):
                    layer = getattr(self, "layer_%d" % i)
                    x = x + layer(F.relu(x, True))
                return x
        
        self.conv0 = torch.nn.Conv2d(self.ni, self.no, 1, padding=0, stride=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(self.ni)
        # self.g_0 = _G(self.no, self.no, self.depth1)
        self.g_left = torch.nn.Linear(self.no, self.no, bias=False)
        self.g_right = torch.nn.Linear(self.no, self.no, bias=False)
        self.g = _G(self.no, self.no, self.depth2)
        self.f = _F(self.no, self.no, self.depth3)

    def forward(self, x_left, x_right, nclasses, nqueries):
        ## nqueries is the total number of queries, not query_size
        x_left = self.g_left(x_left)
        x_right = self.g_right(x_right)
        x = x_left.view(nclasses, 1, self.no) + x_right.view(1, nqueries, self.no)
        del (x_left)
        del (x_right)
        return self.g(x.view(nclasses*nqueries, self.no))
