"""Visualizing various options of the generator.

This script is meant to play with the various knobs of synbols and have a quick rendering. It can be done inside or
outside of the docker.

Usage with docker:
$ synbols /path/to/view_generator.py
-> the file view_generator.png will be saved in the current directory

Usage without docker:
$ python view_generator.py
-> the rendering will be shown through matplotlib but synbols fonts won't be available.

The `plot_dataset` function offers a visualization tool to group by certain attributes for rows and columns. You can
replace by `None`, to disable the grouping of certain axis. Note that if grouping is activated, there might not be an
image specifically generated for each cell, yielding black squares.
"""

from synbols.data_io import pack_dataset
from synbols import generate
from synbols import drawing

from synbols.visualization import plot_dataset
import matplotlib.pyplot as plt

# bg = drawing.Camouflage(stroke_angle=1.)
# bg = drawing.NoPattern()
bg = drawing.MultiGradient(alpha=0.5, n_gradients=2, types=('linear', 'radial'))
# bg = drawing.Gradient(types=('linear',), random_color=drawing.color_sampler(brightness_range=(0.1, 0.9)))

# fg = drawing.Camouflage(stroke_angle=0.5)
# fg = drawing.SolidColor((1, 1, 1))
fg = drawing.Gradient(types=('radial',), random_color=drawing.color_sampler(brightness_range=(0.1, 0.9)))

attr_sampler = generate.basic_image_sampler(resolution=(64, 64), foreground=fg, background=bg)
# attr_sampler = generate.add_occlusion(attr_sampler)
x, mask, y = pack_dataset(generate.dataset_generator(attr_sampler, 500, generate.flatten_mask))

plt.figure('dataset')
plot_dataset(x, y, h_axis='char', v_axis='font', n_row=10, n_col=20)

plt.savefig('view_generator.png')
