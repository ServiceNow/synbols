from torch.utils.data import Dataset
from .synbols import SynbolsNpz
from .episodic_dataset import EpisodicDataset
import numpy as np
import json
import sys
import os
#sys.path.insert(0, '/home/optimass/synbols/benchmarks/') 
#sys.path.insert(0, '../..') 
#sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) 
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
#sys.path.insert(0, os.getcwd())
#print(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) 
# from classification import datasets as cls_dataset
try:
    from benchmarks.classification import datasets as cls_dataset
except:
    #sys.path.insert(0, '/home/optimass/synbols/benchmarks/') 
    sys.path.insert(0, '/home/optimass/synbols/benchmarks/') 
    from classification import datasets as cls_dataset

class EpisodicSynbols(EpisodicDataset):
    def __init__(self, path, split, sampler, size, key='font', transform=None):
        #dataset = SynbolsNpz(path, split, key, transform)
        dataset = cls_dataset.SynbolsHDF5(path, split, key, transform)
        self.x = dataset.x
        self.name = "synbols"
        super().__init__(dataset.y, sampler, size, dataset.transform)

    def sample_images(self, indices):
        return [self.transforms(self.x[i]) for i in indices]

if __name__ == '__main__':
    synbols = Synbols('/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz', 'val')