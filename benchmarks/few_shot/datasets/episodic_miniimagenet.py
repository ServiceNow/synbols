import os
from .episodic_dataset import EpisodicDataset
import numpy as np

class EpisodicMiniImagenet(EpisodicDataset):
    tasks_type = "clss"
    name = "miniimagenet"
    episodic=True
    split_paths = {"train":"train", "valid":"val", "val":"val", "test": "test"}
    # c = 3
    # h = 84
    # w = 84

    def __init__(self, data_root, split, sampler, size, transforms):
        """ Constructor

        Args:
            split: data split
            few_shot_sampler: FewShotSampler instance
            task: dataset task (if more than one)
            size: number of tasks to generate (int)
            disjoint: whether to create disjoint splits.
        """
        self.data_root = os.path.join(data_root, "mini-imagenet-%s.npz")
        self.split = split
        data = np.load(self.data_root % self.split_paths[split])
        self.features = data["features"]
        labels = data["targets"]
        del(data)
        super().__init__(labels, sampler, size, transforms)

    def sample_images(self, indices):
        return self.features[indices]

    def __iter__(self):
        return super().__iter__()