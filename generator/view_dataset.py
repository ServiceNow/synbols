#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from synbols.data_io import load_h5, load_attributes_h5, load_minibatch_h5
import logging
from synbols.utils import flatten_attr
import sys

logging.basicConfig(level=logging.INFO)


def _extract_axis(y, axis_name, max_val):
    if axis_name is None:
        return [None] * max_val

    counter = Counter([attr[axis_name] for attr in y])
    return [e for e, _ in counter.most_common(max_val)]


def plot_dataset(x, y, h_axis='char', v_axis='font', name="dataset", n_row=20, n_col=40, rng=np.random):
    fig = plt.figure(name)
    plt.axis('off')
    plt.tight_layout()
    h_values = _extract_axis(y, h_axis, n_col)
    v_values = _extract_axis(y, v_axis, n_row)
    attr_map = defaultdict(list)
    for i, attr in enumerate(y):
        attr_map[(attr.get(h_axis), attr.get(v_axis))].append(i)

    # h_values = rng.choice(h_values, np.minimum(n_col, len(h_values)), replace=False)
    # v_values = rng.choice(v_values, np.minimum(n_row, len(v_values)), replace=False)

    img_grid = []
    blank_image = np.zeros(x.shape[1:], dtype=x.dtype)

    for v_value in v_values:
        img_row = []
        for h_value in h_values:
            if len(attr_map[(h_value, v_value)]) > 0:

                idx = attr_map[(h_value, v_value)].pop()
                img_row.append(x[idx])
            else:
                img_row.append(blank_image)

        img_grid.append(np.hstack(img_row))

    img_grid = np.vstack(img_grid)

    if img_grid.shape[-1] == 1:
        img_grid = img_grid.squeeze(axis=-1)
        print(img_grid.shape)

    plt.imshow(img_grid)
    plt.xlabel(h_axis)
    plt.ylabel(v_axis)

    fig.tight_layout()


def map_to_class_id(values):
    class_map = {val: i for i, val in enumerate(np.unique(values))}
    class_id = [class_map[val] for val in values]
    return np.array(class_id)


def view_split(split_mask, attr_list, attr_keys, name):
    """Plot an histogram of the marginal for each attribute in attr_keys and each subset specified in split_mask"""

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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = '../partly-occluded_n=10000_2020-May-05.h5py'

    print('load dataset')

    x, mask, attr_list, splits = load_h5(file_path)
    print("x.shape:", x.shape)

    attr_list = [flatten_attr(attr) for attr in attr_list]
    print("list of attributes:")
    for attr in attr_list[0].keys():
        print("  " + attr)

    for split_name, split in splits.items():
        print("making statistics for %s." % split_name)
        view_split(split, attr_list, ['char', 'font', 'scale', 'rotation'],
                   split_name)

    plot_dataset(x, attr_list, v_axis=None, h_axis='font')
    # plt.savefig("dataset.png")
    plt.show()
