#!/usr/bin/env python
import argparse
import logging

logging.basicConfig(level=logging.INFO)

from datetime import datetime
from synbols.fonts import ALPHABET_MAP

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='name of the predefined dataset', default='default')
parser.add_argument('--n_samples', help='number of samples to generate', type=int, default=10000)
parser.add_argument('--alphabet', help='of the alphabet to use', default='default')
parser.add_argument('--resolution', help="""Image resolution e.g.: "32x32". Defaults to the dataset's default.""",
                    default='default')

if __name__ == "__main__":

    args = parser.parse_args()

    from synbols.generate import DATASET_GENERATOR_MAP, make_preview
    from synbols.data_io import write_h5

    for alphabet_name, alphabet in ALPHABET_MAP.items():
        logging.info("%s: %d symbols, %d fonts", alphabet_name, len(alphabet.symbols), len(alphabet.fonts))

    logging.info("Generating %d samples from %s dataset", args.n_samples, args.dataset)

    if args.alphabet == 'default':
        alphabet = 'latin'
        file_path = '%s_n=%d_%s' % (args.dataset, args.n_samples, datetime.now().strftime("%Y-%b-%d"))
    else:
        alphabet = args.alphabet
        file_path = '%s(%s)_n=%d_%s' % (args.dataset, alphabet, args.n_samples, datetime.now().strftime("%Y-%b-%d"))

    ds_generator = DATASET_GENERATOR_MAP[args.dataset](args.n_samples, alphabet=alphabet)
    ds_generator = make_preview(ds_generator, file_path + "_preview.png")
    write_h5(file_path + ".h5py", ds_generator, args.n_samples)
