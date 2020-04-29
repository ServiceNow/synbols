from .active_learning import ActiveLearning, CalibratedActiveLearning, MixUpActiveLearning
from .classification import Classification
from .fewshot import FewShot


def get_model(exp_dict):
    if exp_dict["model"] == 'classification':
        return Classification(exp_dict["num_classes"])
    elif exp_dict["model"] == 'fewshot':
        return FewShot(exp_dict)
    elif exp_dict["model"] == 'active_learning':
        return ActiveLearning(exp_dict)
    elif exp_dict["model"] == 'calibrated_active_learning':
        return CalibratedActiveLearning(exp_dict)
    elif exp_dict["model"] == 'mixup_active_learning':
        return MixUpActiveLearning(exp_dict)
    else:
        raise ValueError("Model %s not found" % exp_dict["model"])
