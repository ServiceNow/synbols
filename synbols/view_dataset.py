import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from data_io import load_npz
import logging

logging.basicConfig(level=logging.INFO)


def _extract_axis(y, h_axis, v_axis):
    h_values = np.unique([attr[h_axis] for attr in y])
    v_values = np.unique([attr[v_axis] for attr in y])

    map = defaultdict(list)

    for i, attr in enumerate(y):
        map[(attr[h_axis], attr[v_axis])].append(i)

    return map, h_values, v_values


def plot_dataset(x, y, h_axis='char', v_axis='font', name="dataset", n_row=20, n_col=30, rng=np.random):
    fig = plt.figure(name)
    # plt.axis('off')

    attr_map, h_values, v_values = _extract_axis(y, h_axis, v_axis)
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
    plt.show()


if __name__ == "__main__":
    x, mask, y = load_npz('../default_n=10000.npz')
    plot_dataset(x, y)
