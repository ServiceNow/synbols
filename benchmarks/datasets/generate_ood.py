from data_io import load_dataset_jpeg_sequential
from collections import defaultdict
import numpy as np

attr_list = list(load_dataset_jpeg_sequential('../../camouflage_n=10000.zip', only_label=True))


def marginalize_attributes(attr_list):
    attr_marginals = defaultdict(list)

    for idx, attr in enumerate(attr_list):
        for key, val in attr.items():

            if isinstance(val, list):
                for i, e in enumerate(val):
                    attr_marginals['%s[%d]' % (key, i)].append((idx, e))
            else:

                attr_marginals[key].append((idx, val))

    return attr_marginals


attr_marginals = marginalize_attributes(attr_list)


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


def unique_class_based_partition(values, sample_ids, ratios):
    """Split according to unique values"""
    unique_values = np.unique(values)
    class_partition = partition_array(unique_values, ratios)

    value_to_partition = {}
    for i, part in enumerate(class_partition):
        for element in part:
            value_to_partition[element] = i

    idx_partition = defaultdict(list)
    for value, id in zip(values, sample_ids):
        part_idx = value_to_partition[value]
        idx_partition[part_idx].append(id)

    return [idx_partition[i] for i in range(len(idx_partition.keys()))]


n = 100
values = np.random.choice(list('abcdefghij'), n)
sample_ids = np.arange(n)
ratios = (0.6, 0.2, 0.2)

# id_partition = unique_class_based_partition(values, sample_ids, ratios)
# for part in id_partition:
#     print([values[id] for id in part])


def quantile_partition(values, sample_ids, ratios):
    pass


values = np.random.randn(100)
# quantile_partition(values, sample_ids, ratios)
print(np.percentile(values, [10, 90]))