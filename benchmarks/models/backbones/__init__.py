import torch
import torchvision.models as models

def get_backbone(exp_dict):
    if exp_dict["backbone"] == "resnet18":
        backbone = models.resnet18(pretrained=exp_dict["imagenet_pretraining"], progress=True)
        return torch.nn.Sequential(*list(backbone.children())[:-2]) #removes last fc
    elif exp_dict["backbone"] == "vgg16":
        backbone = models.vgg16(pretrained=exp_dict["imagenet_pretraining"], progress=True)
        return torch.nn.Sequential(*list(backbone.children())[:-2]) #removes last fc
    else:
        raise(ValueError("Backbone name %s not found" %exp_dict["backbone"]))