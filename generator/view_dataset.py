import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from synbols.data_io import load_npz, load_h5
import logging

logging.basicConfig(level=logging.INFO)


def _extract_axis(y, axis_name, max_val):
    counter = Counter([attr[axis_name] for attr in y])
    return [e for e, _ in counter.most_common(max_val)]


def plot_dataset(x, y, h_axis='char', v_axis='font', name="dataset", n_row=20, n_col=30, rng=np.random):
    fig = plt.figure(name)
    # plt.axis('off')

    h_values = _extract_axis(y, h_axis, n_col)
    v_values = _extract_axis(y, v_axis, n_row)
    attr_map = defaultdict(list)
    for i, attr in enumerate(y):
        attr_map[(attr[h_axis], attr[v_axis])].append(i)

    h_values = rng.choice(h_values, np.minimum(n_col, len(h_values)), replace=False)
    v_values = rng.choice(v_values, np.minimum(n_row, len(v_values)), replace=False)

    img_grid = []
    blank_image = np.zeros(x.shape[1:], dtype=x.dtype)

    for v_value in v_values:
        img_row = []
        for h_value in h_values:
            if (h_value, v_value) in attr_map.keys():

                idx = attr_map[(h_value, v_value)][0]
                img_row.append(x[idx])
            else:
                img_row.append(blank_image)

        img_grid.append(np.hstack(img_row))

    img_grid = np.vstack(img_grid)

    plt.imshow(img_grid)
    plt.xlabel(h_axis)
    plt.ylabel(v_axis)

    fig.tight_layout()


if __name__ == "__main__":
    # x, mask, y = load_npz('../camouflage_n=100000.npz')
    x, mask, y = load_h5('../default_n=2000_2020-Apr-09.h5py')
    print(x.shape)
    plot_dataset(x, y)
    # plt.savefig("dataset.png")
    plt.show()
