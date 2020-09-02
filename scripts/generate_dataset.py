#!/usr/bin/env python
import argparse
import logging
from datetime import datetime
from synbols.generate import DATASET_GENERATOR_MAP, make_preview
from synbols.data_io import write_h5
from synbols.fonts import ALPHABET_MAP

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

dataset_names = ' | '.join(DATASET_GENERATOR_MAP.keys())
alphabet_names = ' | '.join(ALPHABET_MAP.keys())

parser.add_argument('--dataset', help='Name of the predefined dataset. One of %s' % dataset_names, default='default')
parser.add_argument('--n_samples', help='number of samples to generate', type=int, default=10000)
parser.add_argument('--alphabet', help='Which alphabet to use. One of %s' % alphabet_names, default='default')
parser.add_argument('--resolution', help="""Image resolution e.g.: "32x32". Defaults to the dataset's default.""",
                    default='default')

if __name__ == "__main__":

    args = parser.parse_args()

    if args.alphabet == 'default':
        alphabet = 'latin'
        file_path = '%s_n=%d_%s' % (args.dataset, args.n_samples, datetime.now().strftime("%Y-%b-%d"))
    else:
        alphabet = args.alphabet
        file_path = '%s(%s)_n=%d_%s' % (args.dataset, alphabet, args.n_samples, datetime.now().strftime("%Y-%b-%d"))

    ds_generator = DATASET_GENERATOR_MAP[args.dataset](args.n_samples, alphabet=alphabet)
    ds_generator = make_preview(ds_generator, file_path + "_preview.png")
    write_h5(file_path + ".h5py", ds_generator, args.n_samples)
