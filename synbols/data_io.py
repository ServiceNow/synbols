import numpy as np
import json
import zipfile
from PIL import Image
from os import path
import logging


def load_dataset_npz(file_path):
    """Load the dataset from compressed numpy format (npz)."""
    dataset = np.load(file_path)
    y = dataset['y']
    y = [json.loads(attr) for attr in y]
    return dataset['x'], y


def load_dataset_jpeg_sequential(file_path, max_samples=None):
    logging.info("openning %s" % file_path)
    with zipfile.ZipFile(file_path) as zf:

        name_list = zf.namelist()

        x, y = None, None

        n_samples = 0

        for name in name_list:
            with zf.open(name) as fd:

                prefix, ext = path.splitext(name)
                if ext == '.jpeg':
                    img = Image.open(fd)
                    x = np.array(img)
                    prefix_jpeg = prefix

                    if n_samples % 1000 == 0:
                        logging.info("loading sample %d" % n_samples)

                if ext == '.json':
                    y = json.load(fd)
                    prefix_json = prefix

                if x is not None and y is not None:
                    assert prefix_jpeg == prefix_json
                    yield x, y
                    x, y = None, None
                    n_samples += 1
                    if max_samples is not None and n_samples >= max_samples:
                        break


def pack_dataset(generator):
    """Turn a the output of a generator of (x,y) pairs into a numpy array containing the full dataset"""
    x, y = zip(*generator)
    return np.stack(x), y


def write_numpy(file_path, generator):
    x, y = zip(*list(generator()))
    x = np.stack(x)

    print("Saving dataset in %s." % file_path)
    np.savez(file_path, x=x, y=y)


def write_jpg_zip(directory, generator):
    """Write the dataset in a zipped directory using jpeg and json for each image."""
    with zipfile.ZipFile(directory + '.zip', 'w') as zf:
        for i, (x, y) in enumerate(generator()):
            name = "%s/%07d" % (directory, i)
            with zf.open(name + '.jpeg', 'w') as fd:
                Image.fromarray(x).save(fd, 'jpeg', quality=90)
            zf.writestr(name + '.json', y)
