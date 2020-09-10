"""
Example Usage:
````
$ cd /my/dataset/directory
$ synbols /path/to/this/script/joint_attribute_distribution.py
```
"""

from synbols.generate import generate_and_write_dataset, basic_attribute_sampler
import numpy as np
from synbols import drawing

dataset_path = "./joint_distribution_dataset"
n_samples = 100

colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # blue green red
color_distr_dict = {'a': (0.8, 0.2, 0),  # 80% of 'a's are blue and 20% are green
                    'b': (0, 0.8, 0.2),
                    'c': (0.2, 0, 0.8)}


def attribute_sampler():
    """Makes brightness dependent on scale and color dependant on symbol."""

    char = np.random.choice(list('abc'))
    color_index = np.random.choice(len(colors), p=color_distr_dict[char])
    color = colors[color_index]

    scale = 0.6 * np.exp(np.random.randn() * 0.2)

    brightness = 0.4 * np.exp(np.random.randn() * 0.2) / scale

    color = tuple(np.array(color) * brightness)

    fg = drawing.SolidColor(color)

    attr_sampler = basic_attribute_sampler(
        char=char, foreground=fg, background=drawing.NoPattern(), inverse_color=False,
        scale=scale, resolution=(64, 64), max_contrast=False)
    return attr_sampler()  # returns a single sample from this specific attribute sampler


generate_and_write_dataset(dataset_path, attribute_sampler, n_samples)
