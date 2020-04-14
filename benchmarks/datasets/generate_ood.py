import __init__
# from ...generator.synbols import data_io as load_npz
from generator.synbols.data_io import load_npz

import numpy as np


def partition_array(values, ratios):
    """General function to partition an array into sub-arrays of relative size determined by ratios"""
    np.testing.assert_almost_equal(np.sum(ratios), 1., decimal=3)
    n = len(values)
    sizes = np.round(np.array(ratios) * n, 0).astype(np.int)

    partition = []
    current = 0
    for size in sizes:
        partition.append(values[current:(current + size)])
        current += size

    return partition


def unique_class_based_partition(values, ratios):
    """Split according to unique values"""
    unique_values = np.unique(values)
    class_partition = partition_array(unique_values, ratios)

    value_to_partition = {}
    for i, part in enumerate(class_partition):
        for element in part:
            value_to_partition[element] = i

    part_map = np.array([value_to_partition[value] for value in values])

    return part_map


def percentile_partition(values, ratios):
    """Split according to percentiles of values."""

    brackets = np.cumsum(np.concatenate(((0,), ratios))) * 100
    brackets[-1] = 100

    percentiles = np.percentile(values, brackets)

    part_map = np.zeros(len(values))
    for i in range(len(ratios)):
        mask = np.logical_and(values >= percentiles[i], values <= percentiles[i + 1])
        part_map[mask] = i

    return part_map


def split_partition_map(part_map):
    """Return a list of masks for each unique value in part_map"""
    return [part_map == partition_id for partition_id in np.unique(part_map)]


def compositional_split(part_map_1, part_map_2):
    """"""
    partition_ids = np.unique(part_map_1)
    partition_masks = []
    for _ in partition_ids:
        partition_masks.append(part_map_1 == part_map_2)
        part_map_2 = (part_map_2 + 1) % len(partition_ids)
    return partition_masks


if __name__ == "__main__":

    # Example usage and verification of the behavior

    import matplotlib.pyplot as plt

    # attr_list = load_npz('../../default_n=10000.npz')[2]
    attr_list = load_npz('/mnt/datasets/public/research/synbols/default_n=100000.npz')[2]
    # X in [0], mask in [1], attr in [2]

    axis1 = np.array([attr['char'] for attr in attr_list])
    axis2 = np.array([attr['scale'][0] for attr in attr_list])
    #axis2 = np.array([attr['scale'] for attr in attr_list])

    ratios = (0.5, 0.2, 0.3)
    # play with this a little

    part_map_1 = unique_class_based_partition(values=axis1, ratios=ratios)
    part_map_2 = unique_class_based_partition(values=axis2, ratios=ratios)

    print("char split")
    for part_mask in split_partition_map(part_map_1):
        subset = axis1[part_mask]
        print("axis1 len: %d, (%d unique values) %s" % (len(subset), len(np.unique(subset)), np.unique(subset)))

    print("scale split")
    for part_mask in split_partition_map(part_map_2):
        subset = axis2[part_mask]
        print("axis2 len: %d, (min: %.3g, max: %.3g)" % (len(subset), np.min(subset), np.max(subset)))

    print("Compositional split")
    for part_mask in compositional_split(part_map_1, part_map_2):
        subset1 = axis1[part_mask]

        unique_char = np.unique((subset1))
        char_map = dict(zip(*(unique_char, list(range(len(unique_char))))))
        char_id = [char_map[char] for char in subset1]

        subset2 = axis2[part_mask]
        print("axis1 len: %d, (%d unique values)" % (len(subset1), len(np.unique(subset1))))
        print("axis2 len: %d, (min: %.3g, max: %.3g)" % (len(subset2), np.min(subset2), np.max(subset2)))

        plt.plot(char_id, subset2, '.')
        plt.xlabel('char')
        plt.ylabel('scale')

    plt.show()
