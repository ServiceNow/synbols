"""Tools for visualizing current font clusters.

Usage:
$ cd /where/the/png/will/be/saved
$ synbols view_font_clustering.py
"""

import json
from synbols.generate import basic_attribute_sampler
from synbols.drawing import SolidColor
import numpy as np
from synbols.fonts import ALPHABET_MAP
from PIL import Image


def cluster_to_img_grid(font_cluster):
    bg = SolidColor((0, 0, 0))
    fg = SolidColor((1, 1, 1))

    img_grid = []
    for font, _d in font_cluster:
        img_list = []
        for char in 'abcdefghijkl':
            img = basic_attribute_sampler(font=font, char=char, is_bold=False, is_slant=False, scale=1., translation=(0, 0),
                                          background=bg, foreground=fg, rotation=0, inverse_color=False,
                                          resolution=(128, 128))()
            img_list.append(img.make_image())

        img_grid.append(np.hstack(img_list))
    return np.vstack(img_grid)


if __name__ == "__main__":
    from os import path

    print("current number of latin fonts %d" % (len(ALPHABET_MAP['latin'].fonts)))

    with open(path.join(path.dirname(__file__), '../synbols/fonts/hierarchical_clustering_font.json')) as fd:
        clusters = json.load(fd)

    for i, cluster in enumerate(clusters):
        for name, val in cluster:
            print(name, "%.3g" % val)
        print()

        img_grid = cluster_to_img_grid(cluster)

        Image.fromarray(img_grid).save("cluster_%d.png" % i)
