from .classification import Classification
from .fewshot import FewShot
from . import detect, semseg

def get_model(exp_dict):
    if exp_dict["model"] == 'classification':
        return Classification(exp_dict) 
    elif exp_dict["model"] == 'fewshot':
        return FewShot(exp_dict) 
    elif exp_dict["model"] == 'semseg':
        fcn = models.segmentation.fcn_resnet101(pretrained=True).eval().cuda()

        return semseg.SemSeg(base_model=fcn, exp_dict)
         
    elif exp_dict["model"] == 'detect':
        return detect.Detector(exp_dict) 
    else:
        raise ValueError("Model %s not found" %exp_dict["model"])