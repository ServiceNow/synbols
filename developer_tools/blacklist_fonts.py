import subprocess as sp
from glob import glob
import json
import logging
import os

import numpy as np
import h5py
from scipy.cluster.hierarchy import linkage
import torch
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


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def culster_fonts(model_path, data_path, use_gpu=False):
    data = h5py.File(data_path, 'r')
    dataset = Dataset(data)
    data.close()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    logger.info("Loading pytorch model")
    backbone = torch.load(model_path, map_location=torch.device('cpu')).cpu()

    backbone.classifier = Identity()
    backbone.fc = Identity()
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
            features.append(backbone(image).data.cpu().numpy())
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
    clusters = culster_fonts(font_model_path, synbols_default_bw_path)

    logger.info("Saving json")
    with open('font_clusters.json', 'w') as outfile:
        json.dump(clusters, outfile)

