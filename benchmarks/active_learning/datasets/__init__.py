from baal.active import ActiveLearningDataset
from torchvision import transforms as tt

from .aleatoric_uncertainty_datasets import AleatoricSynbols
from .synbols import Synbols


def get_dataset(split, dataset_dict):
    if dataset_dict["name"] == "active_learning":
        transform = tt.Compose([tt.ToPILImage(), tt.ToTensor()])
        n_classes = dataset_dict.get('n_classes')
        n_classes = n_classes or (52 if dataset_dict['task'] == 'char' else 1002)
        dataset = AleatoricSynbols(path=dataset_dict["path"],
                                   split=split,
                                   key=dataset_dict["task"],
                                   transform=transform,
                                   p=dataset_dict.get('p', 0.0),
                                   seed=dataset_dict.get('seed', 666),
                                   n_classes=n_classes,
                                   pixel_sigma=dataset_dict.get('pixel_sigma', 0.0),
                                   pixel_p=dataset_dict.get('pixel_p', 0.0),
                                   )
        if split == 'train':
            dataset = ActiveLearningDataset(dataset, pool_specifics={'transform': transform})
            dataset.label_randomly(dataset_dict['initial_pool'])
        return dataset
    else:
        raise ValueError("Dataset %s not found" % dataset_dict["name"])
