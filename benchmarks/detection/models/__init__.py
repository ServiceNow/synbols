
from . import lcfcn


def get_model(model_dict,  exp_dict, train_set=None):
    name = model_dict['name']


    if name == "lcfcn":
        model =  lcfcn.LCFCN(n_classes=train_set.n_classes).cuda()
        model.opt = torch.optim.Adam(
            (model.parameters()), lr=1e-5)