from torch.utils.data import Dataset
from .synbols import SynbolsNpz
from .episodic_dataset import EpisodicDataset
import numpy as np
import json
import sys
import os
try:
    from benchmarks.classification import datasets as cls_dataset
except:
    sys.path.insert(0, '/home/optimass/synbols/benchmarks/') 
    from classification import datasets as cls_dataset

class EpisodicSynbols(EpisodicDataset):
    def __init__(self, path, split, sampler, size, key='font', transform=None, mask=None):
        if 'npz' in path:
            dataset = SynbolsNpz(path, split, key, transform)
        elif 'h5py' in path:
            dataset = cls_dataset.SynbolsHDF5(path, split, key, transform, mask=mask)
        else:
            Exception('not implemented')
        self.x = dataset.x
        self.name = "synbols"
        super().__init__(dataset.y, sampler, size, dataset.transform)

    def sample_images(self, indices):
        return [self.transforms(self.x[i]) for i in indices]

if __name__ == '__main__':
    synbols = Synbols('/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz', 'val')