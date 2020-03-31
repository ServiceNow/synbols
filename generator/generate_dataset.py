#!/usr/bin/python
import argparse
import logging

from synbols.generate import *


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='name of the predefined dataset', default='default')
parser.add_argument('--n_samples', help='number of samples to generate', type=int, default=10000)

DATASET_GENERATOR_MAP = {
    'plain': generate_plain_dataset,
    'default': generate_default_dataset,
    'camouflage': generate_camouflage_dataset,
    'segmentation': generate_segmentation_dataset,
}

if __name__ == "__main__":
    args = parser.parse_args()

    logging.info("Generating %d samples from %s dataset", args.n_samples, args.dataset)
    ds_generator = DATASET_GENERATOR_MAP[args.dataset](args.n_samples)

    directory = '%s_n=%d' % (args.dataset, args.n_samples)
    write_npz(directory, ds_generator)
