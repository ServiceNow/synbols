import torch
import torchvision.models as models
from torch import nn

def get_backbone(exp_dict):
    if exp_dict["backbone"] == "resnet18":
        backbone = models.resnet18(pretrained=exp_dict["imagenet_pretraining"], progress=True)
        return torch.nn.Sequential(*list(backbone.children())[:-2]) #removes last fc
    if exp_dict["backbone"] == "conv":
        ## only gonna work on Synbols for now
        ## and MAML, because of different output dim

        class Flatten(nn.Module):
            def forward(self, input):
                return input.view(input.size(0), -1)
        
        net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(2*2*64, exp_dict['dataset']['nclasses_train']))
        return net
    else:
        raise(ValueError("Backbone name %s not found" %exp_dict["backbone"]))