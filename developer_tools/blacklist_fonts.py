import json
import logging
import multiprocessing as mp
import os
import subprocess as sp

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tt
from scipy.cluster.hierarchy import linkage
from tqdm import tqdm

from synbols.data_io import write_h5
from synbols.drawing import SolidColor
from synbols.generate import basic_attribute_sampler, LANGUAGE_MAP, dataset_generator

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()

font_classifier_remote_path = ("https://github.com/ElementAI/synbols-resources/raw/master/models/"
                               "font_clustering_feature_extractor_resnet12.pth")

char_classifier_remote_path = ("https://github.com/ElementAI/synbols-resources/raw/master/models/"
                               "font_filtering_char_classifier_resnet12.pth")

def prepare_environment(font_model_remote_path, char_model_remote_path, n_samples=100000, target_dir='/tmp'):
    font_model_path = os.path.join(target_dir, os.path.basename(font_model_remote_path))
    char_model_path = os.path.join(target_dir, os.path.basename(char_model_remote_path))

    sp.run(["wget", "--continue", font_model_remote_path], cwd=target_dir)
    sp.run(["wget", "--continue", char_model_remote_path], cwd=target_dir)

    synbols_default_bw_path = os.path.join(target_dir, "synbols_default-bw_n=%d.h5py" % n_samples)
    if not os.path.exists(synbols_default_bw_path):

        attr_sampler = basic_attribute_sampler(
            alphabet=LANGUAGE_MAP['english'].get_alphabet(auxiliary=False),
            background=SolidColor((0, 0, 0)),
            foreground=SolidColor((1, 1, 1))
        )

        ds_generator = dataset_generator(attr_sampler, n_samples)
        write_h5(synbols_default_bw_path, ds_generator, n_samples)
    else:
        logger.info("Reusing existing dataset %s." % synbols_default_bw_path)

    return font_model_path, char_model_path, synbols_default_bw_path


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
        assert (width == 1)  # Comment for different variants of this model
        self.widths = [x * int(width) for x in [64, 128, 256]]
        self.widths.append(self.output_size * width)
        self.bn_out = torch.nn.BatchNorm1d(self.output_size)
        self.classifier = torch.nn.Linear(self.output_size, nclasses)
        start_width = in_ch
        for i in range(len(self.widths)):
            setattr(self, "group_%d" % i, Block(start_width, self.widths[i], 1, dropout))
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
    def __init__(self, data):
        y = data['y'][...]

        with mp.Pool(8) as pool:
            self.meta = pool.map(json.loads, y)

        y_font = np.array([_y['font'] for _y in self.meta])
        y_char = np.array([_y['char'] for _y in self.meta])
        
        self.x = data['x'][...]
        self.fontset = list(sorted(set(y_font)))
        self.charset = list(sorted(set(y_char)))
        self.fonts = np.zeros(y_font.shape[0], dtype=int)
        self.chars = np.zeros(y_char.shape[0], dtype=int)
        for i, font in enumerate(self.fontset):
            self.fonts[y_font == font] = i

        for i, char in enumerate(self.charset):
            self.chars[y_char == char] = i

        n_channels = 3
        self.transforms = [tt.ToPILImage(), tt.ToTensor(), tt.Normalize([0.5] * n_channels, [0.5] * n_channels)]
        self.transforms = tt.Compose(self.transforms)

    def __getitem__(self, idx):
        return self.transforms(self.x[idx]), self.font[idx], self.char[idx]

    def __len__(self):
        return len(self.x)


def cluster_fonts(model_path, data_path, use_gpu=False):
    data = h5py.File(data_path, 'r')
    dataset = Dataset(data)
    data.close()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    logger.info("Loading pytorch model")
    weights = torch.load(model_path, map_location=torch.device('cpu'))['model']
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
        for image, font, char in tqdm(dataloader):
            if use_gpu:
                image = image.cuda()
            features.append(backbone.up_to_embedding(image).mean(-1).mean(-1).data.cpu().numpy())
            labels.append(font)
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

def filter_fonts(model_path, data_path, use_gpu=False):
    data = h5py.File(data_path, 'r')
    dataset = Dataset(data)
    data.close()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    logger.info("Loading pytorch model")
    weights = torch.load(model_path, map_location=torch.device('cpu'))['model']
    nclasses = weights['classifier.weight'].shape[0]
    backbone = Resnet12(1, 3, nclasses)
    backbone.load_state_dict(weights)

    logits = []
    fonts = []
    chars = []
    logger.info("Extracting predictions")
    with torch.no_grad():
        backbone.eval()
        if use_gpu:
            backbone.cuda()
        for image, font, char in tqdm(dataloader):
            if use_gpu:
                image = image.cuda()
            logits.append(F.softmax(backbone(image), -1).data.cpu().numpy())
            fonts.append(font.numpy())
            chars.append(char.numpy())
    logits = np.concatenate(logits, 0)
    preds = logits.argmax(-1)
    fonts = np.concatenate(fonts)
    chars = np.concatenate(chars)

    error_dict = {font: [] for font in dataset.fontset}

    font2str = {i: dataset.fontset[i] for i in range(len(dataset.fontset))}
    char2str = {i: dataset.charset[i] for i in range(len(dataset.charset))}

    for font, char, pred in zip(fonts, chars, preds):
        font_name = font2str[font]
        char_name = char2str[char]
        pred_name = char2str[int(pred)]
        error_dict[font_name].append((char_name, pred_name))

    return error_dict

def clusters_to_blacklist(clusters):
    blacklist = []
    comments = []
    for cluster in clusters:
        for font, value in cluster[1:]:
            blacklist.append(font)
            comments.append("similar to %s (%.3g)" % (cluster[0][0], value))

    return blacklist, comments


def blacklist_to_tsv(blacklist, comments):
    lines = []
    for font, comment in zip(blacklist, comments):
        lines.append("%s\t%s" % (font, comment))
    return '\n'.join(lines)


if __name__ == "__main__":
    font_model_path, char_model_path, synbols_default_bw_path = prepare_environment(font_classifier_remote_path, 
                                                                                    char_classifier_remote_path,
                                                                                    n_samples=100000)
    clusters = cluster_fonts(font_model_path, synbols_default_bw_path)
    difficult_fonts = filter_fonts(char_model_path, synbols_default_bw_path)

    with open('font_clusters_english.json', 'w') as outfile:
        json.dump(clusters, outfile)

    blacklist_tsv = blacklist_to_tsv(*clusters_to_blacklist(clusters))
    with open("blacklist_english.tsv", 'w') as fd:
        fd.write(blacklist_tsv)
