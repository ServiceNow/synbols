from synbols.generate import generate_and_write_dataset, basic_image_sampler

"""
Example Usage:

$ cd /my/dataset/directory
$ synbols /path/to/this/script/fix_scale_dataset.py

The synbols command will launch the docker containing the fonts (and download it if necessary). It will also
mount the current directory for saving the dataset and execute this script.

Alternatively, you can mount any extra directory with the arguement --mount-path, and have the dataset written at this
location by changing the `dataset_path` variable to point to that location.
"""

dataset_path = "./fix_scale_dataset"
n_samples = 10000

# Modifies the default attribute sampler to fix the scale to 0.8
attr_sampler = basic_image_sampler(scale=0.8)

generate_and_write_dataset(dataset_path, attr_sampler, n_samples)

