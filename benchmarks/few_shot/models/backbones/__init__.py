
from torch import nn

def get_backbone(exp_dict,
                    architecture=None,
                    hidden_size=None,
                    output_size=None,
                    feature_extractor=False):

    ## Boilerplate

    if architecture is None:
        architecture = exp_dict["backbone"]
    
    if hidden_size is None:
        #TODO: might not work for all hidden_size
        hidden_size = int(exp_dict["hidden_size"])
    
    if feature_extractor:
        output_size = hidden_size
    else:
        if output_size is None:
            output_size = int(exp_dict["dataset"]["nclasses_train"])

    ## define model

    if architecture == "resnet18":
        backbone = models.resnet18(pretrained=exp_dict["imagenet_pretraining"], progress=True)
        if feature_extractor:
           return torch.nn.Sequential(*list(backbone.children())[:-2]) #removes last fc
        else:
            #TODO: i guess this will something else to specify the output size
            return torch.nn.Sequential(*list(backbone.children()))
    
    if architecture == "conv4":
        ## only gonna work on Synbols for now

        class Flatten(nn.Module):
            def forward(self, input):
                return input.view(input.size(0), -1)
        
        net = nn.Sequential(
            nn.Conv2d(3, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_size, hidden_size, 3, 1, 1),
            nn.BatchNorm2d(hidden_size, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(2*2*hidden_size, output_size)
        )
        return net
    
    if architecture == "mlp3":
        net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )
        return net
        

    
    else:
        raise(ValueError("Backbone name %s not found" %exp_dict["backbone"]))