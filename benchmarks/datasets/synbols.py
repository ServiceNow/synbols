from torch.utils.data import Dataset
import numpy as np
import json

class Synbols(Dataset):
    def __init__(self, path, split, key='font', transform=None):
        self.path = path
        self.split = split
        data = np.load(path) 
        self.x = data['x']
        self.y = data['y']
        del(data)
        _y = []
        for y in self.y:
            _y.append(json.loads(y)[key])
        self.y = _y
        self.labelset = list(sorted(set(self.y)))
        self.y = np.array([self.labelset.index(y) for y in self.y])
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        self.num_classes = len(self.labelset)
        self.make_splits()

    def make_splits(self, seed=42):
        if self.split == 'train':
            start = 0
            end = int(0.8 * len(self.x))
        elif self.split == 'val':
            start = int(0.8 * len(self.x))
            end = int(0.9 * len(self.x))
        elif self.split == 'test':
            start = int(0.9 * len(self.x))
            end = len(self.x) 
        rng = np.random.RandomState(seed)
        self.indices = rng.permutation(len(self.x))
        self.x = self.x[self.indices[start:end]]
        self.y = self.y[self.indices[start:end]]

    def __getitem__(self, item):
        return self.transform(self.x[item]), self.y[item]

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    synbols = Synbols('/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz', 'val')