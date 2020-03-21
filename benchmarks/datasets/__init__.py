from torchvision import transforms as tt
from .synbols import Synbols
from .episodic_dataset import FewShotSampler # should be imported here because it changes the dataloader to be episodic
from .episodic_synbols import EpisodicSynbols 

def get_dataset(split, exp_dict):
    dataset_dict = exp_dict["dataset"]
    if dataset_dict["name"] == "synbols":
        transform = tt.Compose([tt.ToPILImage(), tt.ToTensor()])
        ret = Synbols(dataset_dict["path"], split, dataset_dict["task"], transform)
        # exp_dict["num_classes"] = ret.num_classes # FIXME: this is hacky
        return ret
    elif dataset_dict["name"] == "fewshot_synbols":
        transform = tt.Compose([tt.ToPILImage(), tt.ToTensor()])
        sampler = FewShotSampler(nclasses=dataset_dict["nclasses_%s" %split],
                                 support_size=dataset_dict["support_size_%s" %split],
                                 query_size=dataset_dict["query_size_%s" %split],
                                 unlabeled_size=0)
        return EpisodicSynbols(dataset_dict["path"], 
                                split=split, 
                                sampler=sampler, 
                                size=dataset_dict["%s_iters" %split], 
                                key=dataset_dict["task"], 
                                transform=transform)
    else:
        raise ValueError("Dataset %s not found" % dataset_dict["name"])
