from baal.active import ActiveLearningDataset
from torchvision import transforms as tt

from .aleatoric_uncertainty_datasets import AleatoricSynbols
from .episodic_dataset import FewShotSampler
from .episodic_synbols import EpisodicSynbols
from .synbols import Synbols


def get_dataset(split, dataset_dict):
    if dataset_dict["name"] == "synbols":
        transform = tt.Compose([tt.ToPILImage(), tt.ToTensor()])
        ret = Synbols(dataset_dict["path"], split, dataset_dict["task"], transform)
        exp_dict["num_classes"] = ret.num_classes  # FIXME: this is hacky
        return ret
    elif dataset_dict["name"] == "fewshot_synbols":
        transform = tt.Compose([tt.ToPILImage(), tt.ToTensor()])
        sampler = FewShotSampler(nclasses=dataset_dict["nclasses_%s" % split],
                                 support_size=dataset_dict["support_size_%s" % split],
                                 query_size=dataset_dict["query_size_%s" % split],
                                 unlabeled_size=0)
        return EpisodicSynbols(dataset_dict["path"],
                               split=split,
                               sampler=sampler,
                               size=dataset_dict["%s_iters" % split],
                               key=dataset_dict["task"],
                               transform=transform)
    elif dataset_dict["name"] == "active_learning":

        transform = tt.Compose([tt.ToPILImage(), tt.ToTensor()])
        dataset = AleatoricSynbols(uncertainty_config=dataset_dict.get('uncertainty_config', {}),
                                   path=dataset_dict["path"],
                                   split=split,
                                   key=dataset_dict["task"],
                                   transform=transform,
                                   p=dataset_dict.get('p', 0.0),
                                   seed=dataset_dict.get('seed', 666),
                                   n_classes=dataset_dict.get('n_classes'),
                                   pixel_sigma = dataset_dict.get('pixel_sigma', 0.0),
                                   pixel_p = dataset_dict.get('pixel_p', 0.0),
                                   self_supervised=dataset_dict.get('self_supervised'))

        if split == 'train':
            dataset = ActiveLearningDataset(dataset, pool_specifics={'transform': transform,
                                                                     'self_supervised': False})
            dataset.label_randomly(dataset_dict['initial_pool'])
        return dataset
    else:
        raise ValueError("Dataset %s not found" % dataset_dict["name"])
