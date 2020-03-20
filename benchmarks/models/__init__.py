from .classification import Classification
from .protonet import protonet

def get_model(exp_dict):
    if exp_dict["benchmark"] == 'classification':
        return Classification(exp_dict) 
    elif exp_dict["benchmark"] == 'fewshot': 
        if exp_dict["model"] == 'protonet':
            return protonet(exp_dict) 
        elif exp_dict["model"] == 'MAML':
            print('not implemented yet')
            return FewShot(exp_dict) 
    else:
        raise ValueError("Model %s not found" %exp_dict["model"])