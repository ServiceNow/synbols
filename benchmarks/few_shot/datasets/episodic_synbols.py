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
    def __init__(self, path, split, sampler, size, key='font', transform=None,
                    mask=None, trim_size=None):
        if 'npz' in path:
            dataset = SynbolsNpz(path, split, key, transform)
        elif 'h5py' in path:
            dataset = cls_dataset.SynbolsHDF5(path, split, key, transform,
                    mask=mask, trim_size=trim_size)
        else:
            Exception('not implemented')
        self.x = dataset.x
        self.name = "synbols"
        ## make sure we have enough data per class:
        unique, counts = np.unique(dataset.y, return_counts=True)
        #TODO: dont harcode this
        low_data_classes = unique[counts < 15] # 5-shot 5-query 
        low_data_classes_idx = np.isin(dataset.y, low_data_classes)
        self.x = self.x[~low_data_classes_idx]
        dataset.y = dataset.y[~low_data_classes_idx] 

        super().__init__(dataset.y, sampler, size, dataset.transform)
    
    def sample_images(self, indices):
        return [self.transforms(self.x[i]) for i in indices]

if __name__ == '__main__':
    synbols = Synbols('/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz', 'val')