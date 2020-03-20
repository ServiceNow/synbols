#!/usr/bin/python


import synbols
import time as t
import cairo
import argparse
from data_io import write_jpg_zip
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='name of the predefined dataset', default='camouflage')
parser.add_argument('--n_samples', help='number of samples to generate', type=int, default=100000)


def attribute_generator(n_samples, **kwargs):
    """Generic attribute generator. kwargs is directly passed to the Attributes constructor."""

    def generator():
        for i in range(n_samples):
            yield synbols.Attributes(**kwargs)

    return generator


def dataset_generator(attr_generator, n_samples):
    """High level function generating the dataset from an attribute generator."""

    t0 = t.time()
    for i, attributes in enumerate(attr_generator):

        x = attributes.make_image()
        y = attributes.attribute_dict()

        if i % 100 == 0 and i != 0:
            dt = (t.time() - t0) / 100.
            eta = (n_samples - i) * dt
            eta_str = t.strftime("%Hh%Mm%Ss", t.gmtime(eta))

            logging.info("generating sample %4d / %d (%.3g s/image) ETA: %s", i, n_samples, dt, eta_str)
            t0 = t.time()
        yield x, y



def generate_char_grid(alphabet_name, n_char, n_font, rng=np.random, **kwargs):
    def _attr_generator():
        alphabet = synbols.ALPHABET_MAP[alphabet_name]

        chars = rng.choice(alphabet.symbols, n_char, replace=False)
        fonts = rng.choice(alphabet.fonts, n_font, replace=False)

        for char in chars:
            for font in fonts:
                yield synbols.Attributes(alphabet, char, font, rng=rng, **kwargs)

    return dataset_generator(_attr_generator(), n_char * n_font)


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


def generate_camouflage_dataset(n_samples):
    alphabet = synbols.ALPHABET_MAP['latin']
    camouflage = synbols.Camouflage()
    attr_generator = attribute_generator(n_samples, alphabet=alphabet, slant=cairo.FontSlant.NORMAL, is_bold=True,
                                         foreground=camouflage, background=camouflage)
    return dataset_generator(attr_generator, n_samples)


DATASET_GENERATOR_MAP = {
    'plain': generate_plain_dataset,
    'default': generate_default_dataset,
    'camouflage': generate_camouflage_dataset,
}

if __name__ == "__main__":
    args = parser.parse_args()
    ds_generator = DATASET_GENERATOR_MAP[args.dataset](args.n_samples)

    directory = '%s_n=%d' % (args.dataset, args.n_samples)
    write_jpg_zip(directory, ds_generator)
