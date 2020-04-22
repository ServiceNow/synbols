from .ProtoNet import ProtoNet
from .MAML import MAML
from .RelationNet import RelationNet

def get_model(exp_dict):
    if exp_dict["benchmark"] == 'fewshot': 
        if exp_dict["model"] == 'ProtoNet':
            return ProtoNet(exp_dict) 
        elif exp_dict["model"] == 'MAML':
            return MAML(exp_dict) 
        elif exp_dict["model"] == 'RelationNet':
            return RelationNet(exp_dict) 
    else:
        raise ValueError("Model %s not found" %exp_dict["model"])