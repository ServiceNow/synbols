from torch.utils.data import Dataset
import numpy as np
import json
import os
from torchvision import transforms as tt
from torchvision.datasets import MNIST, SVHN
from PIL import Image
import torch
import h5py
import multiprocessing


def get_dataset(split, exp_dict):
    dataset_dict = exp_dict["dataset"]
    if dataset_dict["name"] == "synbols_folder":
        transform = []
        if dataset_dict["augmentation"] and split == "train":
            transform += [tt.RandomResizedCrop(size=(dataset_dict["height"], dataset_dict["width"]), scale=(0.8, 1)),
                         tt.RandomHorizontalFlip(),
                         tt.ColorJitter(0.4, 0.4, 0.4, 0.4)]
        transform += [tt.ToTensor(),
                      tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        transform = tt.Compose(transform)
        ret = SynbolsFolder(dataset_dict["path"], split, dataset_dict["task"], transform)
        exp_dict["num_classes"] = len(ret.labelset) # FIXME: this is hacky
        return ret
    elif dataset_dict["name"] == "synbols_npz":
        transform = [tt.ToPILImage()]
        if dataset_dict["augmentation"] and split == "train":
            transform += [tt.RandomResizedCrop(size=(dataset_dict["height"], dataset_dict["width"]), scale=(0.8, 1)),
                         tt.RandomHorizontalFlip(),
                         tt.ColorJitter(0.4, 0.4, 0.4, 0.4)]
        transform += [tt.ToTensor(),
                      tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        transform = tt.Compose(transform)
        ret = SynbolsNpz(dataset_dict["path"], split, dataset_dict["task"], transform)
        exp_dict["num_classes"] = len(ret.labelset) # FIXME: this is hacky
        return ret
    elif dataset_dict["name"] == "synbols_hdf5":
        transform = [tt.ToPILImage()]
        if dataset_dict["augmentation"] and split == "train":
            transform += [tt.RandomResizedCrop(size=(dataset_dict["height"], dataset_dict["width"]), scale=(0.8, 1)),
                         tt.RandomHorizontalFlip(),
                         tt.ColorJitter(0.4, 0.4, 0.4, 0.4)]
        transform += [tt.ToTensor(),
                      tt.Normalize([0.5] * dataset_dict["channels"], 
                                    [0.5] * dataset_dict["channels"])]
        transform = tt.Compose(transform)
        ret = SynbolsHDF5(dataset_dict["path"], split, dataset_dict["task"], transform, mask=dataset_dict["mask"])
        exp_dict["num_classes"] = len(ret.labelset) # FIXME: this is hacky
        return ret
    elif dataset_dict["name"] == "mnist":
        transform = []
        if dataset_dict["augmentation"] and split == "train":
            transform += [tt.RandomResizedCrop(size=(dataset_dict["height"], dataset_dict["width"]), scale=(0.8, 1)),
                         tt.RandomHorizontalFlip()]
        else:
            transform += [tt.Resize(dataset_dict["height"])]
        transform += [tt.ToTensor(),
                      tt.Normalize([0.5], [0.5])]
        transform = tt.Compose(transform)
        ret = MNIST('/mnt/datasets/public/research/pau', train=(split=="train"), transform=transform, download=True)
        exp_dict["num_classes"] = 10 # FIXME: this is hacky
        return ret
    elif dataset_dict["name"] == "svhn":
        transform = []
        if dataset_dict["augmentation"] and split == "train":
            transform += [tt.RandomResizedCrop(size=(dataset_dict["height"], dataset_dict["width"]), scale=(0.8, 1)),
                         tt.RandomHorizontalFlip(),
                         tt.ColorJitter(0.4, 0.4, 0.4, 0.4)]
        transform += [tt.ToTensor(),
                      tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        transform = tt.Compose(transform)
        split_dict = {'train': 'train', 'test': 'test', 'val': 'test'}
        ret = SVHN('/mnt/datasets/public/research', split=split_dict[split], transform=transform, download=True)
        exp_dict["num_classes"] = 10 # FIXME: this is hacky
        return ret
    else:
        raise ValueError

def _read_json_key(args):
    string, key = args
    return json.loads(string)[key]

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
        data = np.load(self.path, allow_pickle=True) 
        self.x = data['x']
        self.y = data['y']
        del(data)
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        if False:
            self.save_load_cache()
        else:
            self.make_splits()

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

    def make_splits(self, mask=None, json=False, seed=42):
        if mask is None:
            if self.split == 'train':
                start = 0
                end = int(self.train_fraction * len(self.x))
            elif self.split == 'val':
                start = int(self.train_fraction * len(self.x))
                end = int((self.train_fraction +self.val_fraction) * len(self.x))
            elif self.split == 'test':
                start = int((self.train_fraction + self.val_fraction) * len(self.x))
                end = len(self.x) 
            rng = np.random.RandomState(seed) #TODO: fix this dangerous thing
            indices = rng.permutation(len(self.x))
            indices = indices[start:end]
        else:
            indices = np.arange(len(self.x)) # 0....nsamples
            indices = indices[mask[:, ["train", "val", "test"].index(self.split)]]
        print("Converting json strings to labels...")
        if json:
            with multiprocessing.Pool(8) as pool:
                _y = pool.map(_read_json_key, zip(self.y, [self.task] * len(self.y)))
        else:
            _y = [y[self.task] for y in self.y]
        print("Done.")
        self.y = _y
        self.labelset = list(sorted(set(self.y)))
        self.y = np.array([self.labelset.index(y) for y in self.y])
        self.x = self.x[indices]
        self.y = self.y[indices]
        # import pylab
        # pylab.hist(self.y, bins=np.arange(len(self.labelset)))
        # pylab.savefig('hist_%s.png' % self.split)
    def __getitem__(self, item):
        return self.transform(self.x[item]), self.y[item]

    def __len__(self):
        return len(self.x)

class SynbolsHDF5(SynbolsNpz):
    def __init__(self, path, split, key='font', transform=None, train_fraction=0.6, val_fraction=0.4, mask=None):
        self.path = path
        self.split = split
        self.task = key
        self.mask = mask
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = 1 - train_fraction - val_fraction
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        print("Loading hdf5...")
        with h5py.File(path, 'r') as data:
            self.x = data['x'][...]
            self.y = data['y'][...]
            if "split" in data:
                if mask is not None:
                    mask = data["split"][mask][...]
            self.make_splits(mask=mask, json=True)

        print("Done.")

if __name__ == '__main__':
    synbols = SynbolsHDF5('/mnt/datasets/public/research/synbols/camouflage_n=100000_2020-Apr-09.h5py', 'val')