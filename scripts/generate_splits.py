#!/usr/bin/env python


from synbols.stratified_splits import make_default_splits
from synbols.data_io import add_splits
import h5py
import json
import logging
import argparse

logging.basicConfig(level=logging.DEBUG)
parser = argparse.ArgumentParser()
parser.add_argument('files', metavar='N', type=str, nargs='+', help='All h5py to be processed')
# parser.add_argument('--ratios', metavar='R', type=float, nargs='+',
#                     help='List of ratios for splits. Default (0.6, 0.2, 0.2)')

parser.add_argument('--clean_old_splits', help="Clean all existing splits before creating new splits",
                    action='store_true')

parser.add_argument('--random_seed', help='Random seed used to generate all splits', type=int, default=42)

if __name__ == "__main__":
    ratios = (0.6, 0.2, 0.2)
    random_seed = 42

    args = parser.parse_args()

    for file_path in args.files:

        logging.info("processing %s", file_path)

        with h5py.File(file_path, 'a') as fd:

            logging.info("current keys : %s", str(fd.keys()))

            if args.clean_old_splits:
                for key in fd.keys():
                    if key.startswith('split'):
                        del fd[key]

            if 'split' in fd.keys():
                ts = fd['split/random'].attrs['timestamp']
                dst = 'split_old_%s' % ts
                logging.info("moving existing split to %s", dst)
                fd.move('split', dst)

            attr_list = [json.loads(attr) for attr in fd['y']]
            split_dict = make_default_splits(attr_list, ratios, random_seed)
            add_splits(fd, random_seed, split_dict)
