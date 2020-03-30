#!/usr/bin/python
import time as t
import cairo
import argparse
from data_io import write_jpg_zip, write_npz
import logging
import numpy as np
import synbols
from google_fonts import ALPHABET_MAP

from synbols.data_io import write_jpg_zip
from synbols.drawing import Attributes, Camouflage
from synbols.fonts import ALPHABET_MAP

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='name of the predefined dataset', default='default')
parser.add_argument('--n_samples', help='number of samples to generate', type=int, default=10000)


# ya basic!
def basic_image_sampler(alphabet=None, char=None, font=None, background=None, foreground=None, is_slant=None,
                        is_bold=None, rotation=None, scale=None, translation=None, inverse_color=None,
                        pixel_noise_scale=0.01, resolution=(32, 32), n_symbols=1, rng=np.random):
    def sampler():
        symbols = []
        for i in range(n_symbols):
            _alphabet = rng.choice(ALPHABET_MAP.values()) if alphabet is None else alphabet
            _char = rng.choice(_alphabet.symbols) if char is None else char
            _font = rng.choice(_alphabet.fonts) if font is None else font
            _is_bold = rng.choice([True, False]) if is_bold is None else is_bold
            _is_slant = rng.choice([True, False]) if is_slant is None else is_slant
            _rotation = rng.randn() * 0.2 if rotation is None else rotation
            _scale = tuple(np.exp(rng.randn(2) * 0.1)) if scale is None else scale

            # TODO the proper amount of translation depends on the scale
            _translation = tuple(rng.rand(2) * 0.6 - 0.3 ) if translation is None else translation
            # _translation = 0.3 * rng.choice([-1, 1], 2) + (0.1, -.1)

            _foreground = synbols.Gradient() if foreground is None else foreground

            symbols.append(synbols.Symbol(
                alphabet=_alphabet, char=_char, font=_font, foreground=_foreground, is_slant=_is_slant,
                is_bold=_is_bold,
                rotation=_rotation, scale=_scale, translation=_translation, rng=rng))

        _background = synbols.Gradient() if background is None else background
        _inverse_color = rng.choice([True, False]) if inverse_color is None else inverse_color

        return synbols.Image(symbols, background=_background, inverse_color=_inverse_color, resolution=resolution,
                             pixel_noise_scale=pixel_noise_scale)

    return sampler


def attribute_generator(sampler, n_samples):
    for i in range(n_samples):
        yield sampler()


def dataset_generator(attr_sampler, n_samples):
    """High level function generating the dataset from an attribute generator."""
    t0 = t.time()
    for i in range(n_samples):
        attributes = attr_sampler()
        mask = attributes.make_mask()
        x = attributes.make_image()
        y = attributes.attribute_dict()

        if i % 100 == 0 and i != 0:
            dt = (t.time() - t0) / 100.
            eta = (n_samples - i) * dt
            eta_str = t.strftime("%Hh%Mm%Ss", t.gmtime(eta))

            logging.info("generating sample %4d / %d (%.3g s/image) ETA: %s", i, n_samples, dt, eta_str)
            t0 = t.time()
        yield x, mask, y


def generate_char_grid(alphabet_name, n_char, n_font, rng=np.random, **kwargs):
    def _attr_generator():
        alphabet = ALPHABET_MAP[alphabet_name]

        chars = rng.choice(alphabet.symbols, n_char, replace=False)
        fonts = rng.choice(alphabet.fonts, n_font, replace=False)

        for char in chars:
            for font in fonts:
                yield basic_image_sampler(alphabet, char, font, rng=rng, **kwargs)()

    return dataset_generator(_attr_generator().__next__, n_char * n_font)


def generate_plain_dataset(n_samples):
    alphabet = ALPHABET_MAP['latin']
    attr_sampler = basic_image_sampler(
        alphabet=alphabet, background=synbols.NoPattern(), foreground=synbols.SolidColor((1, 1, 1,)), is_slant=False,
        is_bold=False, rotation=0, scale=(1., 1.), translation=(0., 0.), inverse_color=False, pixel_noise_scale=0.)
    return dataset_generator(attr_sampler, n_samples)


def generate_default_dataset(n_samples):
    alphabet = ALPHABET_MAP['latin']
    attr_sampler = basic_image_sampler(alphabet=alphabet)
    return dataset_generator(attr_sampler, n_samples)


def generate_camouflage_dataset(n_samples, n_synbols_per_image=1):
    alphabet = ALPHABET_MAP['latin']
    fg = synbols.Camouflage(stroke_angle=0.5)
    bg = synbols.Camouflage(stroke_angle=1.)
    attr_generator = attribute_generator(n_samples, n_synbols_per_image, alphabet=alphabet, is_bold=True, foreground=fg,
                                         background=bg,
                                         scale=(1.3, 1.3))
    return dataset_generator(attr_generator, n_samples, n_synbols_per_image)


def generate_segmentation_dataset(n_samples, n_synbols_per_image=5):
    alphabet = ALPHABET_MAP['latin']
    attr_generator = attribute_generator(n_samples, n_synbols_per_image, alphabet=alphabet,
                                         slant=cairo.FontSlant.NORMAL,
                                         is_bold=False, resolution=(128, 128), background='gradient',
                                         n_symbols_per_image=2, inverse_color=False)
    return dataset_generator(attr_generator, n_samples, n_synbols_per_image)


DATASET_GENERATOR_MAP = {
    'plain': generate_plain_dataset,
    'default': generate_default_dataset,
    'camouflage': generate_camouflage_dataset,
    'segmentation': generate_segmentation_dataset,
}

if __name__ == "__main__":
    args = parser.parse_args()

    logging.info("Generating %d samples from %s dataset", args.n_samples, args.dataset)
    ds_generator = DATASET_GENERATOR_MAP[args.dataset](args.n_samples)

    directory = '%s_n=%d' % (args.dataset, args.n_samples)
    write_npz(directory, ds_generator)
