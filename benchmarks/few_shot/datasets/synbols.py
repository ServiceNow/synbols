from torch.utils.data import Dataset
import numpy as np
import json
import os
from PIL import Image


class SynbolsFolder(Dataset):
    def __init__(self, path, split, key='font', transform=None, train_fraction=0.6, val_fraction=0.4):
        self.path = path
        self.split = split
        self.task = key
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = 1 - train_fraction - val_fraction
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        self.save_load_cache()

    def save_load_cache(self):
        path = "/tmp/%s_%s_%s.npz" %(os.path.basename(self.path), self.task, self.split) 
        if os.path.isfile(path):
            print("Loading %s split from %s..." %(self.split, path))
            data = np.load(path, allow_pickle=True)['arr_0'].item()
            self.x = data['x']
            self.y = data['y']
            self.labelset = list(sorted(set(self.y)))
            print("Done.")
        else:
            print("Saving %s split to %s..." %(self.split, path))
            self.make_splits()
            np.savez(path, {'x': self.x, 'y': self.y})
            print("Done...")

    def make_splits(self, seed=42):
        data = np.load(os.path.join(self.path, 'labels.npz')) 
        self.x = np.array(list(data.keys()))
        self.y = []
        for k in self.x:
            self.y.append(json.loads(str(data[k]))[self.task])
        del(data)
        self.labelset = list(sorted(set(self.y)))
        self.y = np.array([self.labelset.index(y) for y in self.y])
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
        img = Image.open(os.path.join(self.path, 'images', self.x[item] + '.jpeg'))
        return self.transform(img), self.y[item]

    def __len__(self):
        return len(self.x)

class SynbolsNpz(Dataset):
    def __init__(self, path, split, key='font', transform=None, train_fraction=0.6, val_fraction=0.4):
        self.path = path
        self.split = split
        self.task = key
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = 1 - train_fraction - val_fraction
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        self.save_load_cache()

    def save_load_cache(self):
        path = "/tmp/%s_%s_%s.npz" %(os.path.basename(self.path), self.task, self.split) 
        if os.path.isfile(path):
            print("Loading %s split from %s..." %(self.split, path))
            data = np.load(path, allow_pickle=True)['arr_0'].item()
            self.x = data['x']
            self.y = data['y']
            self.labelset = list(sorted(set(self.y)))
            print("Done.")
        else:
            print("Saving %s split to %s..." %(self.split, path))
            self.make_splits()
            np.savez(path, {'x': self.x, 'y': self.y})
            print("Done...")

    def make_splits(self, seed=42):
        data = np.load(self.path, allow_pickle=True) 
        self.x = data['x']
        self.y = data['y']
        del(data)
        _y = []
        for y in self.y:
            _y.append(y[self.task])
        self.y = _y
        self.labelset = list(sorted(set(self.y)))
        self.y = np.array([self.labelset.index(y) for y in self.y])
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

