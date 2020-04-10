import numpy as np
import json
import zipfile
from PIL import Image
from os import path
import logging
import h5py


def load_npz(file_path):
    """Load the dataset from compressed numpy format (npz)."""
    dataset = np.load(file_path, allow_pickle=True)
    return dataset['x'], dataset['mask'], dataset['y']


def write_npz(file_path, generator):
    x, mask, y = zip(*list(generator))
    x = np.stack(x)
    mask = np.stack(mask)

    logging.info("x: %s, %s, mask: %s, %s", x.shape, x.dtype, mask.shape, mask.dtype)
    logging.info("Saving dataset in %s.", file_path)
    np.savez(file_path, x=x, y=y, mask=mask)


class H5Stack:
    def __init__(self, file, name, chunk_size=100, compression="gzip"):
        self.dset = None
        self.file = file
        self.name = name
        self.chunk_size = chunk_size
        self.compression = compression
        self.i = 0

    def add(self, x):

        # create it based on x's information
        if self.dset is None:

            if isinstance(x, str):
                shape = ()
                dtype = h5py.string_dtype(encoding='ascii')
            else:
                shape = x.shape
                dtype = x.dtype

            self.dset = self.file.create_dataset(
                self.name, (self.chunk_size,) + shape, dtype=dtype, maxshape=(None,) + shape,
                chunks=(self.chunk_size,) + shape, compression=self.compression)

        dset = self.dset
        if self.i >= dset.shape[0]:
            dset.resize(dset.shape[0] + self.chunk_size, 0)

        dset[self.i] = x
        self.i += 1


def write_h5(file_path, generator):
    with h5py.File(file_path, 'w', libver='latest') as fd:
        x_stack = H5Stack(fd, 'x')
        mask_stack = H5Stack(fd, 'mask')
        y_stack = H5Stack(fd, 'y')

        for i, (x, mask, y) in enumerate(generator):
            x_stack.add(x)
            mask_stack.add(mask)
            y_stack.add(json.dumps(y))


def load_h5(file_path):
    with h5py.File(file_path, 'r') as fd:
        y = [json.loads(attr) for attr in fd['y']]
        return np.array(fd['x']), np.array(fd['mask']), y


def load_dataset_jpeg_sequential(file_path, max_samples=None):
    logging.info("Opening %s" % file_path)
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
    x, mask, y = zip(*generator)
    return np.stack(x), y


def write_jpg_zip(directory, generator):
    """Write the dataset in a zipped directory using jpeg and json for each image."""
    with zipfile.ZipFile(directory + '.zip', 'w') as zf:
        for i, (x, x2, y) in enumerate(generator):
            name = "%s/%07d" % (directory, i)
            with zf.open(name + '.jpeg', 'w') as fd:
                Image.fromarray(x).save(fd, 'jpeg', quality=90)
            if x2 is not None:
                with zf.open(name + '_gt.jpeg', 'w') as fd:
                    Image.fromarray(x2).save(fd, 'jpeg', quality=90)
            zf.writestr(name + '.json', json.dumps(y))
