# todo(pau) import the necessary code for running this script.

import argparse
import json
import os
import sys
from os.path import join
import numpy as np
import pandas
import pylab
import logging
import scipy
import seaborn as sns
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models
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
args = parser.parse_args()

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()

# This needs to go here, it imports the code to load a dataset
sys.path.insert(0, args.benchmark_code_path)
import datasets

logger.info("Loading dataset")
dataset_info = {'augmentation': False, 
            'height': 32, 
            'name': 'synbols_npz', 
            'path': args.data_path, 
            'task': 'font', 'width': 32, 'channels': 3}
exp_dict = {'dataset':dataset_info}
dataset = datasets.get_dataset("train", exp_dict)

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
loader = DataLoader(dataset, batch_size=512, num_workers=4)
logger.info("Extracting feature embeddings")
with torch.no_grad():
    backbone.eval()
    for image, label in tqdm(loader):
        features.append(backbone(image[...].cpu()).data.cpu().numpy())
        labels.append(label)
features = np.concatenate(features, 0)
labels = np.concatenate([l.numpy() for l in labels], 0)
prototypes = np.zeros((len(np.unique(labels)), features.shape[-1]))
for l in np.unique(labels):
    prototypes[l] = features[labels==l].mean(0)
prototypes = prototypes / np.sqrt((prototypes**2).sum(1, keepdims=True))

logger.info("Doing hierarchical clustering")
Z = scipy.cluster.hierarchy.linkage(prototypes, method='ward', metric='euclidean')

logger.info("Flattening cluster hierarchy")
clusters = []
indices = []

for i, z in enumerate(tqdm(Z[:100])):
    idx = len(confmat) + i
    left, right, score, _ = z
    leafs = []
    if left < len(confmat):
        leafs.append([dataset.labelset[int(left)], score])
    else:
        index = indices.index(left)
        indices.pop(index)
        leafs += clusters.pop(index)
    if right < len(confmat):
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

