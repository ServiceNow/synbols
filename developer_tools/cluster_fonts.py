import argparse
import json
import logging
import os
from os.path import join

import numpy as np
import scipy
import h5py
from scipy.cluster.hierarchy import linkage
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as tt
from tqdm import tqdm
import multiprocessing as mp


# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F

class Block(torch.nn.Module):
    def __init__(self, ni, no, stride, dropout=0, groups=1):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(dropout) if dropout > 0 else lambda x: x
        self.conv0 = torch.nn.Conv2d(ni, no, 3, stride, padding=1, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(no)
        self.conv1 = torch.nn.Conv2d(no, no, 3, 1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(no)
        self.conv2 = torch.nn.Conv2d(no, no, 3, 1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(no)
        if stride == 2 or ni != no:
            self.shortcut = torch.nn.Conv2d(ni, no, 1, stride=1, padding=0)

    def get_parameters(self):
        return self.parameters()

    def forward(self, x, is_support=True):
        y = F.relu(self.bn0(self.conv0(x)), True)
        y = self.dropout(y)
        y = F.relu(self.bn1(self.conv1(y)), True)
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x), True)


class Resnet12(torch.nn.Module):
    def __init__(self, width, in_ch, nclasses, dropout=0.1):
        super().__init__()
        self.output_size = 512
        assert(width == 1) # Comment for different variants of this model
        self.widths = [x * int(width) for x in [64, 128, 256]]
        self.widths.append(self.output_size * width)
        self.bn_out = torch.nn.BatchNorm1d(self.output_size)
        self.classifier = torch.nn.Linear(self.output_size, nclasses)
        start_width = in_ch
        for i in range(len(self.widths)):
            setattr(self, "group_%d" %i, Block(start_width, self.widths[i], 1, dropout))
            start_width = self.widths[i]

    def add_classifier(self, nclasses, name="classifier", modalities=None):
        setattr(self, name, torch.nn.Linear(self.output_size, nclasses))

    def up_to_embedding(self, x, is_support=True):
        """ Applies the four residual groups
        Args:
            x: input images
            n: number of few-shot classes
            k: number of images per few-shot class
            is_support: whether the input is the support set (for non-transductive)
        """
        for i in range(len(self.widths)):
            x = getattr(self, "group_%d" % i)(x, is_support)
            x = F.max_pool2d(x, 3, 2, 1)
        return x

    def forward(self, x):
        """Main Pytorch forward function

        Returns: class logits

        Args:
            x: input mages
            is_support: whether the input is the sample set
        """
        *args, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.up_to_embedding(x, True)
        return self.classifier(F.relu(self.bn_out(x.mean(3).mean(2)), True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--model_path", type=str,
                        default='/mnt/projects/vision_prototypes/synbols/logs_borgy_baselines_june/f46208e10e191f5cee4f5e78a2fe3399',
                        help='Experiment folder, with weights and exp_dict')
    parser.add_argument("-d", "--data_path", type=str,
                        default='/mnt/datasets/public/research/synbols/less-variations_n=100000_2020-May-20.h5py')
    parser.add_argument("-a", "--attribute", type=str, default='font', choices=['char', 'font'], 
                        help="The attribute to cluster")
    parser.add_argument("-o", "--output", type=str,
                        default='./hierarchical_clustering_font.json',
                        help='Output path for the json path')
    parser.add_argument("-t", "--threshold", type=int, default=100,
                        help="Threshold on the distance metric, to stop clustering")
    parser.add_argument("--use_gpu", action='store_true', help="Whether to use GPU")
    args = parser.parse_args()

    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger()

    logger.info("Loading dataset")

    dataset_info = {'augmentation': False,
                    'height': 32,
                    'name': 'synbols_npz',
                    'path': args.data_path,
                    'task': 'font', 'width': 32, 'channels': 3}


    def load_json(x):
        return json.loads(x)[args.attribute]

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data):
            y = data['y'][...]
            with mp.Pool(8) as pool:
                y = np.array(pool.map(load_json, y))
            self.x = data['x'][...]
            self.labelset = list(sorted(set(y)))
            self.y = np.zeros(len(y), dtype=int)
            for i, label in enumerate(self.labelset):
                self.y[y == label] = i
            
            self.transforms = [tt.ToPILImage(), tt.ToTensor(), tt.Normalize([0.5] * dataset_info["channels"],
                                                [0.5] * dataset_info["channels"])]
            self.transforms = tt.Compose(self.transforms)
        
        def __getitem__(self, idx):
            return self.transforms(self.x[idx]), self.y[idx]

        def __len__(self):
            return len(self.x)

    data = h5py.File(args.data_path, 'r')
    dataset = Dataset(data)
    data.close()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    logger.info("Loading pytorch model")
    weights = torch.load(os.path.join(args.model_path, 'model.pth'))['model'] 
    nclasses = weights['classifier.weight'].shape[0]
    backbone = Resnet12(1, 3, nclasses)
    backbone.load_state_dict(weights)

    features = []
    labels = []
    logger.info("Extracting feature embeddings")
    with torch.no_grad():
        backbone.eval()
        if args.use_gpu:
            backbone.cuda()
        for image, label in tqdm(dataloader):
            if args.use_gpu:
                image = image.cuda()
            features.append(backbone.up_to_embedding(image).mean(-1).mean(-1).data.cpu().numpy())
            labels.append(label)
    features = np.concatenate(features, 0)
    labels = np.concatenate([l.numpy() for l in labels], 0)
    prototypes = np.zeros((len(np.unique(labels)), features.shape[-1]))
    for l in np.unique(labels):
        prototypes[l] = features[labels == l].mean(0)
    prototypes = prototypes / np.sqrt((prototypes**2).sum(1, keepdims=True))

    logger.info("Doing hierarchical clustering")
    Z = linkage(
        prototypes, method='ward', metric='euclidean')

    logger.info("Flattening cluster hierarchy")
    clusters = []
    indices = []

    for i, z in enumerate(tqdm(Z[:100])):
        idx = len(prototypes) + i
        left, right, score, _ = z
        leafs = []
        if left < len(prototypes):
            leafs.append([dataset.labelset[int(left)], score])
        else:
            index = indices.index(left)
            indices.pop(index)
            leafs += clusters.pop(index)
        if right < len(prototypes):
            leafs.append([dataset.labelset[int(right)], score])
        else:
            index = indices.index(right)
            indices.pop(index)
            leafs += clusters.pop(index)
        clusters.append(leafs)
        indices.append(idx)

    logger.info("Saving json")
    with open(args.output, 'w') as outfile:
        json.dump(clusters, outfile)
