import matplotlib.pyplot as plt
import numpy as np
import synbols


def plot_dataset(ds, name=None, class_stride=1):
    fig = plt.figure(name)
    plt.axis('off')

    img_grid = []
    for one_class in ds[::class_stride]:
        img_grid.append(np.hstack([x for x in one_class]))
    img_grid = np.vstack(img_grid)
    plt.imshow(img_grid)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = synbols.make_ds_from_lang(synbols.LANGUAGES['latin'], 64, 64, 20)
    plot_dataset(dataset, class_stride=4)
