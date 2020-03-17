from .classification import Classification

def get_model(num_classes, exp_dict):
    if exp_dict["model"] == 'classification':
        return Classification(num_classes, exp_dict) 
    else:
        raise(ValueError("%s does not exist" % exp_dict["model"]))