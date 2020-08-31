"""
Example Usage:
````
$ cd /my/dataset/directory
$ synbols /path/to/this/script/fix_scale_dataset.py
```

The synbols command will launch the docker containing the fonts (and download it if necessary). It will also
mount the current directory for saving the dataset and execute this script.

Alternatively, you can mount any extra directory with the arguement `--mount-path`, and have the dataset written at this
location by changing the `dataset_path` variable to point to that location.
"""

from synbols.generate import generate_and_write_dataset, basic_image_sampler

dataset_path = "./large_translation_dataset"
n_samples = 1000


def translation(rng):
    return tuple(rng.uniform(low=-2, high=2, size=2))


# Modifies the default attribute sampler to fix the scale to a constant and the (x,y) translation to a new distribution
attr_sampler = basic_image_sampler(scale=0.5, translation=translation)

generate_and_write_dataset(dataset_path, attr_sampler, n_samples)

