import numpy as np
from synbols.utils import flatten_attr


def partition_array(values, ratios):
    """General function to partition an array into sub-arrays of \
    relative size determined by ratios
    """
    np.testing.assert_almost_equal(np.sum(ratios), 1.0, decimal=3)
    n = len(values)
    sizes = np.round(np.array(ratios) * n, 0).astype(np.int)
    sizes[0] += n - np.sum(sizes)

    partition = []
    current = 0
    for size in sizes:
        partition.append(values[current : (current + size)])
        current += size

    return partition


def _ratio_to_sizes(n, ratios):
    np.testing.assert_almost_equal(np.sum(ratios), 1.0, decimal=3)
    sizes = np.round(np.array(ratios) * n, 0).astype(np.int)
    sizes[0] += n - np.sum(sizes)
    return sizes


def random_map(n, ratios, rng=np.random):
    sizes = _ratio_to_sizes(n, ratios)
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


# TODO the code could be simpler with sort
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


def make_default_splits(attr_list, ratios, random_seed):
    random_masks = partition_map_to_mask(random_map(len(attr_list), ratios, np.random.RandomState(random_seed)))
    verify_part_mask(random_masks, len(attr_list), ratios)

    stratified_char_map = stratified_split(attr_list, "char", ratios, np.random.RandomState(random_seed))
    stratified_char = partition_map_to_mask(stratified_char_map)
    # TODO we're getting some error from CI here, where stratified_char.shape[1] = 1 instead of 3
    verify_part_mask(stratified_char, len(attr_list), ratios, verify_ratios=False)

    stratified_font_map = stratified_split(attr_list, "font", ratios, np.random.RandomState(random_seed))
    stratified_font = partition_map_to_mask(stratified_font_map)
    verify_part_mask(stratified_font, len(attr_list), ratios, verify_ratios=False)

    compositional_char_font = compositional_split(stratified_char_map, stratified_font_map)
    verify_part_mask(compositional_char_font, len(attr_list), ratios, verify_ratios=False)

    return dict(
        random=random_masks,
        stratified_char=stratified_char,
        stratified_font=stratified_font,
        compositional_char_font=compositional_char_font,
    )


def verify_part_mask(part_mask, n, ratios, verify_ratios=True):
    np.testing.assert_equal(part_mask.shape, (n, len(ratios)))

    if verify_ratios:
        ratios_ = np.mean(part_mask, axis=0)
        np.testing.assert_almost_equal(ratios_, ratios, decimal=2)

    sum = np.sum(part_mask, axis=1)
    np.testing.assert_equal(sum, np.ones_like(sum))


def str_to_id(values):
    convert = False
    if isinstance(values, list):
        if isinstance(values[0], str):
            convert = True
    elif isinstance(values, np.ndarray):
        if values.dtype.type == np.str_:
            convert = True

    if convert:
        unique_vals = np.unique(values)
        val_map = dict(zip(*(unique_vals, list(range(len(unique_vals))))))
        return np.array([val_map[val] for val in values]), convert
    else:
        return values, convert


def plot_split_2d(masks, attr_x, attr_y, name_x="x", name_y="y", mask_names=None):
    for i, mask in enumerate(masks.T):
        print("    %d, (%.1f%%)" % (np.sum(mask), np.mean(mask) * 100))
        if mask_names is None:
            mask_name = "mask %d" % i
        else:
            mask_name = mask_names[i]
        plt.plot(attr_x[mask], attr_y[mask], ".", markersize=2, alpha=1, label=mask_name)

    plt.xlabel(name_x)
    plt.ylabel(name_y)
    legend = plt.legend(
        loc="best",
        scatterpoints=1,
    )

    for lengend_item in legend.legendHandles:
        lengend_item._legmarker.set_markersize(6)


def make_stratified_split(attr_list, axis_name, ratios):
    values = np.array([attr[axis_name] for attr in attr_list])
    values, is_str = str_to_id(values)
    if is_str:
        part_map = unique_class_based_partition(values=values, ratios=ratios, rng=np.random)
    else:
        part_map = percentile_partition(values=values, ratios=ratios)

    masks = partition_map_to_mask(part_map)

    verify_part_mask(masks, len(attr_list), ratios)

    # for mask in masks.T:
    #     subset = values[mask]
    # print("    len: %d (%.1f%%), (min: %.3g, max: %.3g)" % (
    #     len(subset), np.mean(mask) * 100, np.min(subset), np.max(subset)))

    return values, part_map


def make_compositional_split(attr_list, axis1_name, axis2_name, ratios):
    _, part_map_1 = make_stratified_split(attr_list, axis1_name, ratios)
    _, part_map_2 = make_stratified_split(attr_list, axis2_name, ratios)
    return compositional_split(part_map_1, part_map_2)


if __name__ == "__main__":
    # Example usage and verification of the behavior
    from synbols.data_io import load_attributes_h5
    import matplotlib.pyplot as plt

    attr_list, _ = load_attributes_h5("../../default_n=10000_2020-May-21.h5py")
    attr_list = [flatten_attr(attr) for attr in attr_list]
    # ratios = (0.15, 0.05, 0.6, 0.05, 0.15)
    ratios = (0.2, 0.6, 0.2)
    split_names = ("Validation", "Train", "Test")

    axis1_name = "Scale"
    axis2_name = "Rotation"

    axis1, part_map_1 = make_stratified_split(attr_list, axis1_name.lower(), ratios)
    axis2, part_map_2 = make_stratified_split(attr_list, axis2_name.lower(), ratios)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    plt.sca(ax1)
    plt.title("Stratified %s" % axis1_name)
    plot_split_2d(partition_map_to_mask(part_map_1), axis1, axis2, axis1_name, axis2_name, split_names)

    plt.sca(ax2)
    plt.title("Stratified %s" % axis2_name)
    plot_split_2d(partition_map_to_mask(part_map_2), axis1, axis2, axis1_name, axis2_name, split_names)
    plt.ylabel("")

    print("Compositional split")

    compositioanl_masks = compositional_split(part_map_1, part_map_2)
    compositioanl_masks = compositioanl_masks[:, [1, 0, 2]]

    plt.sca(ax3)
    plt.title("Compositional %s-%s" % (axis1_name, axis2_name))
    plot_split_2d(compositioanl_masks, axis1, axis2, axis1_name, axis2_name, split_names)
    plt.ylabel("")

    plt.show()
