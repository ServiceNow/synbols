import subprocess as sp
from glob import glob
import json
import logging
import os

import numpy as np
import h5py
from scipy.cluster.hierarchy import linkage
import torch
import torch.nn.functional as F
import torchvision.transforms as tt
from tqdm import tqdm
import multiprocessing as mp

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()

font_classifier_remote_path = "https://github.com/ElementAI/synbols-resources/raw/master/models/font_clustering_feature_extractor.pth"


def prepare_environment(font_model_remote_path, n_samples=100000, target_dir='/tmp'):
    font_model_path = os.path.join(target_dir, os.path.basename(font_model_remote_path))

    sp.run(["wget", "--continue", font_model_remote_path], cwd=target_dir)

    pattern = '*default-bw*n=%d*h5py' % n_samples

    if len(glob(os.path.join(target_dir, pattern))) == 0:
        sp.run(["synbols-datasets", "--dataset=default-bw", "--n_samples=%d" % n_samples], cwd=target_dir)

    synbols_default_bw_path = glob(os.path.join(target_dir, pattern))[0]

    return font_model_path, synbols_default_bw_path


def load_json(json_str):
    return json.loads(json_str)['font']


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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, attribute='font'):
        y = data['y'][...]

        with mp.Pool(8) as pool:
            y = pool.map(load_json, y)

        y = np.array(y)

        self.x = data['x'][...]
        self.labelset = list(sorted(set(y)))
        self.y = np.zeros(y.shape[0], dtype=int)
        for i, label in enumerate(self.labelset):
            self.y[y == label] = i

        n_channels = 3
        self.transforms = [tt.ToPILImage(), tt.ToTensor(), tt.Normalize([0.5] * n_channels, [0.5] * n_channels)]
        self.transforms = tt.Compose(self.transforms)

    def __getitem__(self, idx):
        return self.transforms(self.x[idx]), self.y[idx]

    def __len__(self):
        return len(self.x)

def cluster_fonts(model_path, data_path, use_gpu=False):
    data = h5py.File(data_path, 'r')
    dataset = Dataset(data)
    data.close()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    logger.info("Loading pytorch model")
    weights = torch.load(model_path, map_location=torch.device('cpu')).cpu()
    nclasses = weights['classifier.weight'].shape[0]
    backbone = Resnet12(1, 3, nclasses)
    backbone.load_state_dict(weights)

    features = []
    labels = []
    logger.info("Extracting feature embeddings")
    with torch.no_grad():
        backbone.eval()
        if use_gpu:
            backbone.cuda()
        for image, label in tqdm(dataloader):
            if use_gpu:
                image = image.cuda()
            features.append(backbone.up_to_embedding(image).mean(-1).mean(-1).data.cpu().numpy())
            labels.append(label)
    features = np.concatenate(features, 0)
    labels = np.concatenate([l.numpy() for l in labels], 0)
    prototypes = np.zeros((len(np.unique(labels)), features.shape[-1]))

    for l in np.unique(labels):
        prototypes[l] = features[labels == l].mean(0)

    logger.info("Doing hierarchical clustering")
    Z = linkage(prototypes, method='ward', metric='euclidean')

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

    clusters = [c for c in clusters if len(c) > 1]

    return clusters


if __name__ == "__main__":
    font_model_path, synbols_default_bw_path = prepare_environment(font_classifier_remote_path)
    clusters = cluster_fonts(font_model_path, synbols_default_bw_path)

    logger.info("Saving json")
    with open('font_clusters.json', 'w') as outfile:
        json.dump(clusters, outfile)

