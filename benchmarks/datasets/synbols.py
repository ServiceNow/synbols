from torch.utils.data import Dataset
import numpy as np
import json

class Synbols(Dataset):
    def __init__(self, path, split, key='font', transform=None, train_fraction=0.6, val_fraction=0.4):
        self.path = path
        self.split = split
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = 1 - train_fraction - val_fraction
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
            end = int(self.train_fraction * len(self.x))
        elif self.split == 'val':
            start = int(self.train_fraction * len(self.x))
            end = int((self.train_fraction +self.val_fraction) * len(self.x))
        elif self.split == 'test':
            start = int((self.train_fraction + self.val_fraction) * len(self.x))
            end = len(self.x) 
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.x))
        self.x = self.x[indices[start:end]]
        self.y = self.y[indices[start:end]]

    def __getitem__(self, item):
        return self.transform(self.x[item]), self.y[item]

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    synbols = Synbols('/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz', 'val')