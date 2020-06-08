import json

import h5py
import numpy as np
from torch.utils.data import Dataset


class Synbols(Dataset):
    def __init__(self, path, split, key='font', transform=None):
        self.path = path
        self.split = split
        self.x, self.y = self._load_data(path)
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

    def _load_data(self, f):
        if f.endswith('h5py'):
            with h5py.File(f, 'r') as f:
                return f['x'].value, f['y'].value
        else:
            data = np.load(f)
            return data['x'], data['y']

    def make_splits(self, seed=42):
        start, end = self.get_splits(self.x)
        rng = np.random.RandomState(seed)
        self.indices = rng.permutation(len(self.x))
        self.x = self.x[self.indices[start:end]]
        self.y = self.y[self.indices[start:end]]

    def get_splits(self, source):
        if self.split == 'train':
            start = 0
            end = int(0.8 * len(source))
        elif self.split == 'val':
            start = int(0.8 * len(source))
            end = int(0.9 * len(source))
        elif self.split == 'test':
            start = int(0.9 * len(source))
            end = len(source)
        return start, end

    def __getitem__(self, item):
        return self.transform(self.x[item]), self.y[item]

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    synbols = Synbols('/mnt/datasets/public/research/synbols/old/latin_res=32x32_n=100000.npz',
                      'val')
