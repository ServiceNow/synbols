import torch
import numpy as np
import torchvision.models as models
# from .efficientnet_pytorch import EfficientNet
# from .warn.models.wide_resnet_cifar_attention import WideResNetAttention
# from .warn.models.wide_resnet_cifar import WideResNet
import torch.nn.functional as F

  
import torch.nn as nn


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Conv4(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=1, height_dim=32, width_dim=32, hid_dim=64, z_dim=64,
                 classify=False, output_size=10, z_dim_multiplier=None):
        super(Conv4, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.classify = classify
        if z_dim_multiplier is not None:
            self.z_dim_multiplier = int(z_dim_multiplier)
        if self.classify:
            self.output_layer = nn.Linear(self.z_dim_multiplier*hid_dim, output_size)

    def forward(self, x):
        x = self.encoder(x)
        out = x.view(x.size(0), -1)
        if self.classify:
            out = self.output_layer(out)
        return out

class MLP(torch.nn.Module):
    def __init__(self, ni, no, nhidden, depth, flatten=True):
        super().__init__()
        self.depth = depth
        self.flatten = flatten
        for i in range(depth):
            if i == 0:
                setattr(self, "linear%d" %i, torch.nn.Linear(ni, nhidden))
            else:
                setattr(self, "linear%d" %i, torch.nn.Linear(nhidden, nhidden))
        if depth == 0:
            nhidden = ni
        self.out = torch.nn.Linear(nhidden, no)

    def forward(self, x):
        if self.flatten:
            x = x.view(x.size(0), -1)
        for i in range(self.depth):
            linear = getattr(self, "linear%d" %i)
            x = F.leaky_relu(linear(x))
        return self.out(x)

def get_backbone(exp_dict, classify=True):

    # Seed
    # -----------
    if 'seed' in exp_dict:
        torch.manual_seed(exp_dict['seed'])
    else:
        torch.manual_seed(42)

    backbone_name = exp_dict["backbone"]["name"].lower()
    if backbone_name == "resnet18":
        backbone = models.resnet18(pretrained=exp_dict["backbone"]["imagenet_pretraining"], progress=True)
        if not classify:
           return torch.nn.Sequential(*list(backbone.children())[:-2]) #removes last fc
        else:
            num_ftrs = backbone.fc.in_features
            backbone.fc = torch.nn.Linear(num_ftrs, nclasses) 
            if exp_dict["dataset"]["channels"] != 3:
                assert(not(exp_dict["backbone"]["imagenet_pretraining"]))
                backbone._modules['conv1'] = torch.nn.Conv2d(exp_dict["dataset"]["channels"], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            return backbone
    elif backbone_name == "resnet50":
        backbone = models.resnet50(pretrained=exp_dict["backbone"]["imagenet_pretraining"], progress=True)
        num_ftrs = backbone.fc.in_features
        backbone.fc = torch.nn.Linear(num_ftrs, nclasses) 
        if exp_dict["dataset"]["channels"] != 3:
            assert(not(exp_dict["backbone"]["imagenet_pretraining"]))
            backbone._modules['conv1'] = torch.nn.Conv2d(exp_dict["dataset"]["channels"], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        return backbone
    elif backbone_name == "warn":
        backbone = WideResNetAttention(28, 10, nclasses, 0, 3, 4, reg_w=0.001,
                 attention_type="softmax")
        if exp_dict["dataset"]["channels"] != 3:
            backbone._modules['conv0'] = torch.nn.Conv2d(exp_dict["dataset"]["channels"], 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        return backbone
    elif backbone_name == "wrn":
        backbone = WideResNet(28, 10, nclasses)
        if exp_dict["dataset"]["channels"] != 3:
            backbone._modules['conv0'] = torch.nn.Conv2d(exp_dict["dataset"]["channels"], 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        return backbone
    elif backbone_name == "vgg16":
        backbone = models.vgg16_bn(pretrained=exp_dict["backbone"]["imagenet_pretraining"], progress=True)
        if exp_dict["dataset"]["channels"] != 3:
            assert(not(exp_dict["backbone"]["imagenet_pretraining"]))
            backbone._modules['features'][0] = torch.nn.Conv2d(exp_dict["dataset"]["channels"], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        children = list(backbone.children())
        children = children[:-2]
        output = []
        output.append(GAP())
        output.append(torch.nn.Linear(512, 4096))
        output.append(torch.nn.ReLU(True))
        output.append(torch.nn.Linear(4096, 4096))
        output.append(torch.nn.ReLU(True))
        output.append(torch.nn.Linear(4096, nclasses))
        output = torch.nn.Sequential(*output)
        children.append(output)
        return torch.nn.Sequential(*children)
    elif backbone_name == "conv4":
        return Conv4(x_dim=exp_dict["dataset"]["channels"],
                     height_dim=exp_dict["dataset"]["height"],
                     width_dim=exp_dict["dataset"]["width"],
                     hid_dim=exp_dict["backbone"]["hidden_size"],
                     z_dim=exp_dict["backbone"]["hidden_size"],
                     classify=classify,
                     output_size=exp_dict["dataset"]["nclasses_train"],
                     z_dim_multiplier=exp_dict["dataset"]["z_dim_multiplier"]
                     )
    elif backbone_name == "efficientnet":
        net = EfficientNet.from_pretrained(exp_dict["backbone"]["type"], num_classes=nclasses)
        def weights_init(m):
            try:
                if not isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.xavier_uniform_(m.weight)
                    torch.nn.init.zeros_(m.bias)
                else:
                    m.reset_running_stats()
                    m.reset_parameters()
            except:
                pass

        net.apply(weights_init)
        return net

    elif backbone_name == "mlp":
        return MLP(ni=exp_dict["dataset"]["height"] * exp_dict["dataset"]["width"] * exp_dict["dataset"]["channels"],
                   no=output_size,
                   nhidden=exp_dict["backbone"]["hidden_size"],
                   depth=exp_dict["backbone"]["depth"])
    else:
        raise ValueError


def count_parameters(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nNumber of parameters in the model: {}\n'.format(n_params))
    return n_params




## OLD but could be useful

# class GAP(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         return x.mean(3).mean(2)

# class Conv4(torch.nn.Module):
#     def __init__(self, in_h, in_w, channels, output_size, gap=True, feature_extractor=False):
#         super().__init__()
#         in_size = min(in_w, in_h)
#         ratio = in_size // 4
#         if ratio >= 16:
#             self.strides = [2, 2, 2, 2]
#         elif ratio >= 8:
#             self.strides = [1, 2, 2, 2]
#         elif ratio >= 4:
#             self.strides = [1, 2, 2, 1]
#         elif ratio >= 2:
#             self.strides = [1, 2, 1, 1]
#         else:
#             self.strides = [1, 1, 1, 1]
#         self.channels = [32, 64, 128, 256]
#         in_ch = channels
#         for i in range(4):
#             setattr(self, "conv%d" %i, torch.nn.Conv2d(in_ch, self.channels[i], 3, self.strides[i], 1, bias=False))
#             setattr(self, "bn%d" %i, torch.nn.BatchNorm2d(self.channels[i]))
#             in_ch = self.channels[i]
#         self.gap = gap
#         if gap:
#             self.out = torch.nn.Linear(self.channels[-1], output_size)
#         else:
#             self.out = torch.nn.Linear(self.channels[-1] * 4 * 4, output_size)
#         self.feature_extractor = False

#     def forward(self, x):
#         for i in range(4):
#             conv = getattr(self, "conv%d" %i)
#             bn = getattr(self, "bn%d" %i)
#             x = conv(x)
#             x = F.leaky_relu(bn(x), inplace=True)
#         if self.gap:
#             out = x.mean(3).mean(2)
#         else:
#             out = x.view(x.size(0), -1)
#         if not self.feature_extractor:
#             out = self.out(out)
#         return out