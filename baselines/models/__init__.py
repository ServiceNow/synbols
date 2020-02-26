from .trainer import Trainer

def get_model(num_classes, exp_dict):
    if exp_dict["model"] == 'trainer':
        return Trainer(num_classes, exp_dict) 