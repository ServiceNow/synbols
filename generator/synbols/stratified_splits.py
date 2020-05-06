import numpy as np


def partition_array(values, ratios):
    """General function to partition an array into sub-arrays of relative size determined by ratios"""
    np.testing.assert_almost_equal(np.sum(ratios), 1., decimal=3)
    n = len(values)
    sizes = np.round(np.array(ratios) * n, 0).astype(np.int)
    sizes[0] += n - np.sum(sizes)

    partition = []
    current = 0
    for size in sizes:
        partition.append(values[current:(current + size)])
        current += size

    return partition


def random_map(n, ratios, rng=np.random):
    np.testing.assert_almost_equal(np.sum(ratios), 1., decimal=3)
    sizes = np.round(np.array(ratios) * n, 0).astype(np.int)
    sizes[0] += n - np.sum(sizes)
    part_map = np.concatenate([[i] * ni for i, ni in enumerate(sizes)])
    rng.shuffle(part_map)
    return part_map


def unique_class_based_partition(values, ratios, rng=None):
    """Split according to unique values"""
    unique_values = np.unique(values)

    if rng is not None:
        rng.shuffle(unique_values)

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


def partition_map_to_mask(part_map):
    """Return a list of masks for each unique value in part_map"""
    return np.stack([part_map == partition_id for partition_id in np.unique(part_map)]).T


def compositional_split(part_map_1, part_map_2):
    """Use 2 partition maps to create compositional split"""
    partition_ids = np.unique(part_map_1)
    partition_masks = []
    for _ in partition_ids:
        partition_masks.append(part_map_1 == part_map_2)
        part_map_2 = (part_map_2 + 1) % len(partition_ids)

    return np.stack(partition_masks).T


def stratified_split(attr_list, attr_name, ratios, rng=np.random):
    values = np.array([attr[attr_name] for attr in attr_list])
    part_map = unique_class_based_partition(values, ratios, rng)
    # part_masks = partition_map_to_mask(part_map)
    return part_map


def compositional_split_ratios_search(attr_list, attr_name_1, attr_name_2, ratios, rng):
    pass


def make_default_splits(attr_list, ratios, random_seed):
    random_masks = partition_map_to_mask(random_map(len(attr_list), ratios, np.random.RandomState(random_seed)))
    verify_part_mask(random_masks, len(attr_list), ratios)

    stratified_char_map = stratified_split(attr_list, 'char', ratios, np.random.RandomState(random_seed))
    stratified_char = partition_map_to_mask(stratified_char_map)
    verify_part_mask(stratified_char, len(attr_list), ratios, verify_ratios=False)

    stratified_font_map = stratified_split(attr_list, 'font', ratios, np.random.RandomState(random_seed))
    stratified_font = partition_map_to_mask(stratified_font_map)
    verify_part_mask(stratified_font, len(attr_list), ratios, verify_ratios=False)

    compositional_char_font = compositional_split(stratified_char_map, stratified_font_map)
    verify_part_mask(compositional_char_font, len(attr_list), ratios, verify_ratios=False)

    return dict(random=random_masks,
                stratified_char=stratified_char,
                stratified_font=stratified_font,
                compositional_char_font=compositional_char_font)


def verify_part_mask(part_mask, n, ratios, verify_ratios=True):
    np.testing.assert_equal(part_mask.shape, (n, len(ratios)))

    if verify_ratios:
        ratios_ = np.mean(part_mask, axis=0)
        np.testing.assert_almost_equal(ratios_, ratios, decimal=2)

    sum = np.sum(part_mask, axis=1)
    np.testing.assert_equal(sum, np.ones_like(sum))


if __name__ == "__main__":

    # Example usage and verification of the behavior
    from synbols.data_io import load_h5
    import matplotlib.pyplot as plt

    attr_list = load_h5('../../default_n=2000_2020-Apr-09.h5py')[2]

    axis1 = np.array([attr['char'] for attr in attr_list])
    axis2 = np.array([attr['scale'] for attr in attr_list])

    ratios = (0.5, 0.2, 0.3)

    part_map_1 = unique_class_based_partition(values=axis1, ratios=ratios)
    part_map_2 = unique_class_based_partition(values=axis2, ratios=ratios)

    print("char split")
    for part_mask in partition_map_to_mask(part_map_1):
        subset = axis1[part_mask]
        print("axis1 len: %d, (%d unique values) %s" % (len(subset), len(np.unique(subset)), np.unique(subset)))

    print("scale split")
    for part_mask in partition_map_to_mask(part_map_2):
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
