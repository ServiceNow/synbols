import argparse
import json
import logging
import os
from os.path import join

import numpy as np
import scipy
from scipy.cluster.hierarchy import linkage
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as tt
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-cp", "--benchmark_code_path", type=str,
                    default='/mnt/projects/vision_prototypes/synbols/font_plain_borgy/c0160f3a00fbe81b5594720c8b21d352/code',
                    help='Classification code path')
parser.add_argument("-e", "--model_path", type=str,
                    default='/mnt/datasets/public/research/pau/synbols/plain_feature_extractor.pth',
                    help='Experiment folder, with weights and exp_dict')
parser.add_argument("-d", "--data_path", type=str,
                    default='/mnt/datasets/public/research/synbols/old/plain_n=1000000.npz')
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
data = np.load(args.data_path, allow_pickle=True)
dataset_info = {'augmentation': False,
                'height': 32,
                'name': 'synbols_npz',
                'path': args.data_path,
                'task': 'font', 'width': 32, 'channels': 3}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        y = np.array([d['font'] for d in data['y']])
        self.x = data['x']
        self.labelset = list(sorted(set(y)))
        self.y = np.zeros(y.shape[0], dtype=int)
        for i, label in enumerate(self.labelset):
            self.y[y == label] = i
        
        self.transforms = [tt.ToPILImage(), tt.ToTensor(), tt.Normalize([0.5] * dataset_info["channels"],
                                            [0.5] * dataset_info["channels"])]
        self.transforms = tt.Compose(self.transforms)
    
    def __getitem__(self, idx):
        return self.transforms(self.x[idx]), self.y[idx]

    def __len__(self):
        return len(self.x)

dataset = Dataset(data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

logger.info("Loading pytorch model")
backbone = torch.load(args.model_path).cpu()

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

backbone.classifier = Identity()
backbone.fc = Identity()
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
        features.append(backbone(image).data.cpu().numpy())
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
