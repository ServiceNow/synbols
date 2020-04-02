import time as t
import cairo
import logging
import numpy as np

from .drawing import Camouflage, Gradient, Image, NoPattern, SolidColor, Symbol
from .fonts import ALPHABET_MAP


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
            _translation = tuple(rng.rand(2) * 0.6 - 0.3) if translation is None else translation
            # _translation = 0.3 * rng.choice([-1, 1], 2) + (0.1, -.1)

            _foreground = Gradient() if foreground is None else foreground

            symbols.append(Symbol(alphabet=_alphabet, char=_char, font=_font, foreground=_foreground,
                                  is_slant=_is_slant, is_bold=_is_bold, rotation=_rotation, scale=_scale,
                                  translation=_translation, rng=rng))

        _background = Gradient() if background is None else background
        _inverse_color = rng.choice([True, False]) if inverse_color is None else inverse_color

        return Image(symbols, background=_background, inverse_color=_inverse_color, resolution=resolution,
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
            # print("generating sample %4d / %d (%.3g s/image) ETA: %s"%(i, n_samples, dt, eta_str))
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
        alphabet=alphabet, background=NoPattern(), foreground=SolidColor((1, 1, 1,)), is_slant=False,
        is_bold=False, rotation=0, scale=(1., 1.), translation=(0., 0.), inverse_color=False, pixel_noise_scale=0.)
    return dataset_generator(attr_sampler, n_samples)


def generate_default_dataset(n_samples):
    alphabet = ALPHABET_MAP['latin']
    attr_sampler = basic_image_sampler(alphabet=alphabet)
    return dataset_generator(attr_sampler, n_samples)


def generate_camouflage_dataset(n_samples):
    alphabet = ALPHABET_MAP['latin']
    fg = Camouflage(stroke_angle=0.5)
    bg = Camouflage(stroke_angle=1.)
    attr_sampler = basic_image_sampler(
        alphabet=alphabet, background=bg, foreground=fg, is_slant=False, is_bold=True, scale=(13., 1.3))

    return dataset_generator(attr_sampler, n_samples)


def generate_segmentation_dataset(n_samples, n_synbols_per_image=5):
    alphabet = ALPHABET_MAP['latin']

    attr_generator = basic_image_sampler(alphabet=alphabet, resolution=(128, 128),
                                         background=Gradient(), n_symbols=n_synbols_per_image)
    return dataset_generator(attr_generator, n_samples)


DATASET_GENERATOR_MAP = {
    'plain': generate_plain_dataset,
    'default': generate_default_dataset,
    'camouflage': generate_camouflage_dataset,
    'segmentation': generate_segmentation_dataset,
}
