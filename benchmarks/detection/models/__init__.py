
from . import semseg
import torch

def get_model(model_dict, exp_dict=None, train_set=None, savedir=None):
    if model_dict['name'] in ["semseg"]:
        model =  semseg.SemSeg(exp_dict, train_set, savedir=savedir)

    return model

