from torch.utils.data import Dataset
import numpy as np
import json
import os
from haven import haven_utils as hu
from torchvision import transforms as tt
from PIL import Image
import glob, torch
import h5py, json


def get_dataset(split, exp_dict):
    dataset_dict = exp_dict["dataset"]
    if dataset_dict["name"] == "detection":
        dataset = DetectionFolder(split)
        return dataset

    
class DetectionFolder(Dataset):
    def __init__(self, split):
        path_base = '/mnt/datasets/public/research/synbols'
        meta_fname = os.path.join('/mnt/datasets/public/issam/data/synbols', 'meta_list_1000.json')
        self.path = os.path.join(path_base, 'segmentation_n=100000_2020-May-04.h5py')
        # meta = load_attributes_h5(self.path)
        # meta_list = meta[0]
        # meta_list = meta_list[:1000]
        # hu.save_pkl(meta_fname, meta_list)
        # self.transform = None
        # load_minibatch_h5(self.path, [indices])
        # self.img_list = glob.glob(self.path+"/*.jpeg")
        self.meta_list = hu.load_pkl(meta_fname)
        symbol_dict = {}
        for i in range (len(self.meta_list)):
            meta =  json.loads(self.meta_list[i])
            for s in meta['symbols']:
                if s['char'] not in symbol_dict:
                    symbol_dict[s['char']] = []
                symbol_dict[s['char']] += [i]

        self.n_classes = 2
        self.symbol_dict = symbol_dict
     
    def __getitem__(self, index):
        img, mask = load_minibatch_h5(self.path, [index])
        meta=  json.loads(self.meta_list[index])
        meta['index'] = index
        char_id_list = [i+1 for i, s in enumerate(meta['symbols']) if s['char'] == 'X']

        mask_out = np.zeros(mask.shape)

        for char_id in char_id_list:
            mask_out[mask == char_id] = 1

        # hu.save_image('tmp.png', mask)
        return {'images': torch.FloatTensor(hu.l2f(img)).squeeze().float()/255., 'masks': torch.FloatTensor(mask_out).float().squeeze(), 'meta':meta}
    def __len__(self):
        return len(self.meta_list)

def load_minibatch_h5(file_path, indices):
    with h5py.File(file_path, 'r') as fd:
        x = np.array(fd['x'][indices])
        mask = np.array(fd['mask'][indices])
    return x, mask

def load_attributes_h5(file_path):
    with h5py.File(file_path, 'r') as fd:
        # y = [json.loads(attr) for attr in fd['y']]
        y = list(fd['y'])
        splits = {}
        if 'split' in fd.keys():
            for key in fd['split'].keys():
                splits[key] = np.array(fd['split'][key])

        return y, splits