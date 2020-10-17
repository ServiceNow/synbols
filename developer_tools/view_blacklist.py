"""Tools for visualizing current font clusters.

Usage:
$ cd /where/the/png/will/be/saved
$ synbols view_font_clustering.py
"""

import json
import synbols
from synbols.generate import basic_attribute_sampler
from synbols.drawing import SolidColor
import numpy as np
from PIL import Image
import os


def cluster_to_img_grid(font_cluster):
    bg = SolidColor((0, 0, 0))
    fg = SolidColor((1, 1, 1))

    img_grid = []
    for font, _d in font_cluster:
        img_list = []
        for char in 'abcdefghijkl':
            img = basic_attribute_sampler(font=font, char=char, is_bold=False, is_slant=False, scale=1.,
                                          translation=(0, 0),
                                          background=bg, foreground=fg, rotation=0, inverse_color=False,
                                          resolution=(128, 128))()
            img_list.append(img.make_image())

        img_grid.append(np.hstack(img_list))
    return np.vstack(img_grid)


def view_clusters(font_cluster_path):
    with open(font_cluster_path) as fd:
        clusters = json.load(fd)

    for i, cluster in enumerate(clusters):
        for name, val in cluster:
            print(name, "%.3g" % val)
        print()

        names = [name for name, val in cluster]

        img_grid = cluster_to_img_grid(cluster)

        Image.fromarray(img_grid).save("%s.png" % ('_'.join(names)))


def view_difficult_fonts(difficult_font_path):
    bg = SolidColor((0, 0, 0))
    fg = SolidColor((1, 1, 1))
    img_grid = []

    with open(difficult_font_path) as fd:
        errors = json.load(fd)

        for font, err_list in errors.items():
            if len(err_list) < 20:
                continue
            print("%s: %s" % (font, str(err_list)))
            img = basic_attribute_sampler(font='arial', char=font, is_bold=False, is_slant=False, scale=1.,
                                          translation=(0, 0),
                                          background=bg, foreground=fg, rotation=0, inverse_color=False,
                                          resolution=(264, 64))()
            img_list = [img.make_image()]

            for true, pred in err_list[:20]:
                img = basic_attribute_sampler(font=font, char=true, is_bold=False, is_slant=False, scale=1.,
                                              translation=(0, 0),
                                              background=bg, foreground=fg, rotation=0, inverse_color=False,
                                              resolution=(64, 64))()
                img_list.append(img.make_image())
            img_grid.append(np.hstack(img_list))
    img_grid = np.vstack(img_grid)

    Image.fromarray(img_grid).save("difficult_chars.png")


if __name__ == "__main__":
    blacklist_dir = os.path.join(os.path.dirname(synbols.__file__), 'fonts', 'blacklist')
    font_cluster_path = os.path.join(blacklist_dir, 'font_clusters_english.json')
    difficult_font_path = "difficult_fonts.json"

    view_difficult_fonts(difficult_font_path)
    # view_clusters(font_cluster_path)