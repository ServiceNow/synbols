from torch.utils.data import Dataset
import numpy as np
import json
import os
from torchvision import transforms as tt
from PIL import Image


def get_dataset(split, exp_dict):
    dataset_dict = exp_dict["dataset"]
    if dataset_dict["name"] == "detection":
        dataset = DetectionFolder(split)
        return dataset

class DetectionFolder(Dataset):
    def __init__(self, split):
        self.path = '/mnt/datasets/public/issam/synbols/segmentation_n=100'
        self.transform = None
     
    def __getitem__(self, item):
        return self.transform(self.x[item]), self.y[item]

    def __len__(self):
        return len(self.x)
