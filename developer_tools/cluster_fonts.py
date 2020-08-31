# todo(pau) import the necessary code for running this script.

from backbones import get_backbone
import models
import datasets
import argparse
import json
import os
import sys
from os.path import join

import haven.haven_utils as hu
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
                    default='/home/pau/git/synbols/benchmarks/classification',
                    help='Classification code path')
parser.add_argument("-e", "--experiment_folder", type=str,
                    default='font_plain/c0160f3a00fbe81b5594720c8b21d352',
                    help='Experiment folder, with weights and exp_dict')
parser.add_argument("-o", "--output", type=str,
                    default='./hierarchical_clustering_font.json',
                    help='Output path for the json path')
parser.add_argument("-t", "--threshold", type=int, default=100,
                    help="Threshold on the distance metric, to stop clustering")
args = parser.parse_args()

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()

sys.path.insert(0, args.benchmark_code_path)

logger.info("Loading dataset")
exp_dict = hu.load_json(join(args.experiment_folder, "exp_dict.json"))
dataset = datasets.get_dataset("train", exp_dict)
exp_dict["dataset"]["channels"] = 3

logger.info("Loading pytorch model")
backbone = get_backbone(exp_dict).cuda()
backbone.load_state_dict(torch.load(os.path.join(
    args.experiment_folder, 'model.pth'))['model'])
logits = []
labels = []
loader = DataLoader(dataset, batch_size=512, num_workers=4)
logger.info("Extracting feature embeddings")
with torch.no_grad():
    backbone.eval()
    for image, label in tqdm(loader):
        logits.append(backbone(image[...].cuda()).data.cpu().numpy())
        labels.append(label)
logits = np.concatenate(logits, 0)
labels = np.concatenate([l.numpy() for l in labels], 0)
confmat = np.zeros((len(np.unique(labels)), len(np.unique(labels))))
total = np.zeros((len(np.unique(labels)), len(np.unique(labels))))
logger.info("Creating confusion matrix")
for logit, label in zip(logits, labels):
    confmat[label, ...] = torch.log(F.softmax(torch.from_numpy(logit))).numpy()
    total[label, ...] += 1
confmat /= total
confmat = confmat / np.sqrt(confmat.sum(-1, keepdims=True)**2)

logger.info("Doing hierarchical clustering")
Z = scipy.cluster.hierarchy.linkage(confmat, method='ward', metric='euclidean')

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

