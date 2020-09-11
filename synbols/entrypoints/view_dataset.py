#!/usr/bin/env python
"""Script for visualizing dataset statistics."""
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
import time


logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--attr_keys", nargs='+', type=str)
    parser.add_argument("--split_name", type=str)
    parser.add_argument("--save", default=None, type=str)
    return parser.parse_args()


def map_to_class_id(values):
    class_map = {val: i for i, val in enumerate(np.unique(values))}
    class_id = [class_map[val] for val in values]
    return np.array(class_id)


def view_split(split_mask, attr_list, attr_keys, name):
    """Plot an histogram of the marginal for each attribute in attr_keys and each subset specified
    in split_mask.
    Args:
        split_mask (ndarray): An array of the masks for a subset (a split) of dataset.
        attr_list (List): List of the split attributes.
        attr_keys (List): List of the attributes that you want to visualize the dataset stats for.
        name (str): The name of split that you want to visualize its attributes.
    """

    n_mask = split_mask.shape[1]

    ratios = ', '.join(['%.1f%%' % (r * 100) for r in split_mask.mean(axis=0)])

    fig, ax_grid = plt.subplots(n_mask, len(attr_keys), sharex='col', num="split %s, split ratios=%s" % (name, ratios))

    for j, attr_key in enumerate(attr_keys):
        print('computing histogram for attr %s' % attr_key)
        values = np.array([attr[attr_key] for attr in attr_list])

        if values.dtype.type is np.str_:
            values = map_to_class_id(values)
            n_bins = np.unique(values) + 1
        else:
            n_bins = 20

        for i in range(n_mask):
            sub_values = values[split_mask[:, i]]
            ax = ax_grid[i, j]
            if i + 1 == n_mask:
                ax.set_xlabel(attr_key)

            ax.hist(sub_values, bins=n_bins)


def main():
    # XXX: Imports are here so that they are done inside the docker image (synbols [...])
    from synbols.data_io import load_h5
    from synbols.utils import flatten_attr
    from synbols.visualization import plot_dataset

    args = parse_args()
    if args.data is None:
        raise Exception(" path to the data is not defined."
                        "Please use `--data` and indicate the path.")
    print('load dataset in h5 format ...')

    x, mask, attr_list, splits = load_h5(args.data)
    print("x.shape:", x.shape)

    attr_list = [flatten_attr(attr) for attr in attr_list]

    all_attr_keys = []
    for attr in attr_list[0].keys():
        all_attr_keys.append(attr)
    check = all(key in all_attr_keys for key in args.attr_keys)

    if not check:
        raise Exception("One or more of the provided attribute keys are not valid."
                        f"The complete list of keys is {all_attr_keys}")

    split_masks = splits[args.split_name]
    print("making statistics for %s." % args.split_name)
    view_split(split_masks, attr_list, args.attr_keys, args.split_name)

    plot_dataset(x, attr_list, v_axis=None, h_axis='font')

    if args.save:
        plt.savefig(args.save)

    plt.show()


def entrypoint():
    subprocess.call(["synbols", __file__] + sys.argv[1:])


if __name__ == "__main__":
    main()
