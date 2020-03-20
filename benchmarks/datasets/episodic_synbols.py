from torch.utils.data import Dataset
from .synbols import Synbols
from .episodic_dataset import EpisodicDataset
import numpy as np
import json

class EpisodicSynbols(EpisodicDataset):
    def __init__(self, path, split, sampler, size, key='font', transform=None):
        dataset = Synbols(path, split, key, transform)
        self.x = dataset.x
        self.name = "synbols"
        super().__init__(dataset.y, sampler, size, dataset.transform)

    def sample_images(self, indices):
        return [self.transforms(self.x[i]) for i in indices]

if __name__ == '__main__':
    synbols = Synbols('/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz', 'val')