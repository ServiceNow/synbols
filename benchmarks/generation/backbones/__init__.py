import torch
import numpy as np
import torchvision.models as models
from .biggan_vae import VAE
import torch.nn.functional as F

def get_backbone(exp_dict):
    # nclasses = exp_dict["num_classes"]
    backbone_name = exp_dict["backbone"]["name"].lower()
    if backbone_name == "biggan":
        backbone = VAE(exp_dict)
        return backbone
    else:
        raise ValueError

