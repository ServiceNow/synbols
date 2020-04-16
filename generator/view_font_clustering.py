import json
from synbols.generate import basic_image_sampler
from synbols.drawing import SolidColor
import numpy as np
from matplotlib import pyplot as plt


def cluster_to_img_grid(font_cluster):
    bg = SolidColor((0, 0, 0))
    fg = SolidColor((1, 1, 1))

    img_grid = []
    for font, _d in font_cluster:
        img_list = []
        for char in 'abcdefghijkl':
            img = basic_image_sampler(font=font, char=char, is_bold=False, is_slant=False, scale=1., translation=(0, 0),
                                      background=bg, foreground=fg, rotation=0, inverse_color=False,
                                      resolution=(128, 128))()
            img_list.append(img.make_image())

        img_grid.append(np.hstack(img_list))
    return np.vstack(img_grid)


if __name__ == "__main__":
    from os import path

    with open(path.join(path.dirname(__file__), 'synbols/fonts/hierarchical_clustering_font.json')) as fd:
        clusters = json.load(fd)

    for i, cluster in enumerate(clusters):
        for name, val in cluster:
            print(name, "%.3g" % val)
        print()

        img_grid = cluster_to_img_grid(cluster)
        plt.figure()
        plt.imshow(img_grid)
        plt.savefig("cluster_%d.png" % i, dpi=200)
