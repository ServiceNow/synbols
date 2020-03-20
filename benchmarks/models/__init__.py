from .classification import Classification
from .fewshot import FewShot

def get_model(exp_dict):
    if exp_dict["model"] == 'classification':
        return Classification(exp_dict) 
    elif exp_dict["model"] == 'fewshot':
        return FewShot(exp_dict) 
    else:
        raise ValueError("Model %s not found" %exp_dict["model"])