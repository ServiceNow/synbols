#!/usr/bin/env python


from synbols.stratified_splits import unique_class_based_partition, split_partition_map, random_map
import h5py
import json
import numpy as np
from datetime import datetime
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


def stratified_split(attr_list, attr_name, ratios, rng=np.random):
    values = np.array([attr[attr_name] for attr in attr_list])
    part_map = unique_class_based_partition(values, ratios, rng)
    part_masks = np.stack(split_partition_map(part_map)).T

    verify_part_mask(part_masks, len(attr_list), ratios, verify_ratios=False)

    return part_masks


def verify_part_mask(part_mask, n, ratios, verify_ratios=True):
    np.testing.assert_equal(part_mask.shape, (n, len(ratios)))

    if verify_ratios:
        ratios_ = np.mean(part_mask, axis=0)
        np.testing.assert_almost_equal(ratios_, ratios, decimal=2)

    sum = np.sum(part_mask, axis=1)
    np.testing.assert_equal(sum, np.ones_like(sum))


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

            part_map = random_map(len(attr_list), ratios, np.random.RandomState(random_seed))
            random_masks = np.stack(split_partition_map(part_map)).T
            verify_part_mask(random_masks, len(attr_list), ratios)
            random_split_ds = fd.create_dataset("split/random", data=random_masks)
            random_split_ds.attrs['timestamp'] = datetime.now().strftime("%Y-%b-%d_%H:%M:%S")
            random_split_ds.attrs['seed'] = random_seed

            stratified_char = stratified_split(attr_list, 'char', ratios, np.random.RandomState(random_seed))
            fd.create_dataset("split/stratified_char", data=stratified_char).attrs['seed'] = random_seed

            stratified_font = stratified_split(attr_list, 'font', ratios, np.random.RandomState(random_seed))
            sf_ds = fd.create_dataset("split/stratified_font", data=stratified_font).attrs['seed'] = random_seed
