#!/usr/bin/python


import numpy as np
import synbols
import time as t
import cairo
import argparse
from PIL import Image
from os import path
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='name of the predefined dataset', default='default')
parser.add_argument('--n_samples', help='number of samples to generate', type=int, default=1000000)


# def alphabet_specific_generator(alphabet, resolution=(32, 32), n_samples=100000, rng=np.random):
#     def generator():
#         for i in range(n_samples):
#             yield synbols.Attributes(alphabet, resolution=resolution, rng=rng)
#
#     return generator


def attribute_generator(n_samples, **kwargs):
    """Generic attribute generator. kwargs is directly passed to the Attributes constructor."""

    def generator():
        for i in range(n_samples):
            yield synbols.Attributes(**kwargs)

    return generator


def dataset_generator(attr_generator, n_samples):
    def generator():
        t0 = t.time()
        for i, attributes in enumerate(attr_generator()):

            x = attributes.make_image()
            y = attributes.to_json()

            if i % 100 == 0 and i != 0:
                dt = (t.time() - t0) / 100.
                eta = (n_samples - i) * dt
                eta_str = t.strftime("%Hh%Mm%Ss", t.gmtime(eta))

                print("generating sample %4d / %d (%.3g s/image) ETA: %s" % (i, n_samples, dt, eta_str))
                t0 = t.time()
            yield x, y

    return generator


def write_numpy(file_path, generator):
    x, y = zip(*list(generator()))
    x = np.stack(x)

    print("Saving dataset in %s." % file_path)
    np.savez(file_path, x=x, y=y)


def write_jpg_zip(directory, generator):
    with zipfile.ZipFile(directory + '.zip', 'w') as zip:
        for i, (x, y) in enumerate(generator()):
            name = "%s/%07d" % (directory, i)
            with zip.open(name + '.jpeg', 'w') as fd:
                Image.fromarray(x).save(fd, 'jpeg')
            zip.writestr(name + '.json', y)


def generate_plain_dataset(n_samples):
    alphabet = synbols.ALPHABET_MAP['latin']
    attr_generator = attribute_generator(n_samples, alphabet=alphabet, background=None, foreground=None,
                                         slant=cairo.FontSlant.NORMAL, is_bold=False, rotation=0, scale=(1., 1.),
                                         translation=(0., 0.), inverse_color=False, pixel_noise_scale=0.)
    return dataset_generator(attr_generator, n_samples)


def generate_default_dataset(n_samples):
    alphabet = synbols.ALPHABET_MAP['latin']
    attr_generator = attribute_generator(n_samples, alphabet=alphabet, slant=cairo.FontSlant.NORMAL, is_bold=False)
    return dataset_generator(attr_generator, n_samples)


DATASET_GENERATOR_MAP = {
    'plain': generate_plain_dataset,
    'default': generate_default_dataset,
}

if __name__ == "__main__":
    args = parser.parse_args()
    ds_generator = DATASET_GENERATOR_MAP[args.dataset](args.n_samples)

    directory = '%s_n=%d' % (args.dataset, args.n_samples)
    write_jpg_zip(directory, ds_generator)
