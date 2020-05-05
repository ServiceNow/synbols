import time as t
import logging
import numpy as np

from .drawing import Camouflage, Gradient, Image, NoPattern, SolidColor, Symbol, MultiGradient
from .fonts import ALPHABET_MAP


def _select(default, value, rng):
    if value is None:
        return default
    elif callable(value):
        return value(rng)
    else:
        return value


# ya basic!
def basic_image_sampler(alphabet=None, char=None, font=None, background=None, foreground=None, is_slant=None,
                        is_bold=None, rotation=None, scale=None, translation=None, inverse_color=None,
                        pixel_noise_scale=0.01, resolution=(32, 32), is_gray=False, n_symbols=1, rng=np.random):
    def sampler():
        symbols = []
        _n_symbols = _select(1, n_symbols, rng)
        for i in range(_n_symbols):
            _alphabet = _select(rng.choice(list(ALPHABET_MAP.values())), alphabet, rng)
            _char = _select(rng.choice(_alphabet.symbols), char, rng)
            _font = _select(rng.choice(_alphabet.fonts), font, rng)
            _is_bold = _select(rng.choice([True, False]), is_bold, rng)
            _is_slant = _select(rng.choice([True, False]), is_slant, rng)
            _rotation = _select(rng.randn() * 0.3, rotation, rng)
            _scale = _select(0.6 * np.exp(rng.randn() * 0.2), scale, rng)
            _translation = _select(tuple(rng.rand(2) * 1.8 - 0.9), translation, rng)
            _foreground = _select(Gradient(), foreground, rng)

            symbols.append(Symbol(alphabet=_alphabet, char=_char, font=_font, foreground=_foreground,
                                  is_slant=_is_slant, is_bold=_is_bold, rotation=_rotation, scale=_scale,
                                  translation=_translation, rng=rng))

        _background = _select(Gradient(), background, rng)
        _inverse_color = _select(rng.choice([True, False]), inverse_color, rng)

        return Image(symbols, background=_background, inverse_color=_inverse_color, resolution=resolution,
                     pixel_noise_scale=pixel_noise_scale, is_gray=is_gray)

    return sampler


def flatten_mask(masks):
    flat_mask = np.zeros(masks.shape[:-1])

    for i in range(masks.shape[-1]):
        flat_mask[(masks[:, :, i] > 2)] = i + 1
    return flat_mask


def dataset_generator(attr_sampler, n_samples, mask_aggregator=None):
    """High level function generating the dataset from an attribute generator."""
    t0 = t.time()
    for i in range(n_samples):
        attributes = attr_sampler()
        mask = attributes.make_mask()
        if mask_aggregator is not None:
            mask = mask_aggregator(mask)
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
                yield basic_image_sampler(alphabet, char=char, font=font, rng=rng, **kwargs)()

    return dataset_generator(_attr_generator().__next__, n_char * n_font, flatten_mask)


def generate_plain_dataset(n_samples, alphabet='latin', **kwargs):
    alphabet = ALPHABET_MAP[alphabet]
    attr_sampler = basic_image_sampler(
        alphabet=alphabet, background=NoPattern(), foreground=SolidColor((1, 1, 1,)), is_slant=False,
        is_bold=False, rotation=0, scale=1., translation=(0., 0.), inverse_color=False, pixel_noise_scale=0.)
    return dataset_generator(attr_sampler, n_samples)


def generate_tiny_dataset(n_samples, alphabet='latin', **kwarg):
    fg = SolidColor((1, 1, 1))
    bg = SolidColor((0, 0, 0))
    attr_sampler = basic_image_sampler(alphabet=ALPHABET_MAP[alphabet], background=bg, foreground=fg, is_bold=False,
                                       is_slant=False, scale=1, resolution=(8, 8), is_gray=True)
    return dataset_generator(attr_sampler, n_samples)


def generate_default_dataset(n_samples, alphabet='latin', **kwarg):
    attr_sampler = basic_image_sampler(alphabet=ALPHABET_MAP[alphabet])
    return dataset_generator(attr_sampler, n_samples)


def generate_camouflage_dataset(n_samples, alphabet='latin', **kwarg):
    def attr_sampler():
        angle = np.random.rand() * np.pi * 2
        fg = Camouflage(stroke_angle=angle)
        bg = Camouflage(stroke_angle=angle + 2 + np.random.randn() * 0.5)
        scale = 0.7 * np.exp(np.random.randn() * 0.1)
        return basic_image_sampler(
            alphabet=ALPHABET_MAP[alphabet], background=bg, foreground=fg, is_bold=True,
            scale=scale)()

    return dataset_generator(attr_sampler, n_samples)


def generate_segmentation_dataset(n_samples, alphabet='latin', resolution=(128, 128), **kwarg):
    def scale(rng):
        return 0.1 * np.exp(rng.randn() * 0.4)

    def n_symbols(rng):
        return rng.choice(list(range(3, 20)))

    attr_generator = basic_image_sampler(alphabet=ALPHABET_MAP[alphabet], resolution=resolution, scale=scale,
                                         is_bold=False, n_symbols=n_symbols)
    return dataset_generator(attr_generator, n_samples, flatten_mask)


def generate_couting_dataset(n_samples, alphabet='latin', resolution=(128, 128), **kwarg):
    def scale(rng):
        return 0.1 * np.exp(rng.randn() * 0.4)

    def n_symbols(rng):
        return rng.choice(list(range(3, 20)))

    def char_sampler(rng):
        if rng.rand() < 0.3:
            return rng.choice(ALPHABET_MAP[alphabet].symbols)
        else:
            return 'k'

    attr_generator = basic_image_sampler(alphabet=ALPHABET_MAP[alphabet], char=char_sampler, resolution=resolution,
                                         scale=scale, is_bold=False, n_symbols=n_symbols)
    return dataset_generator(attr_generator, n_samples, flatten_mask)


def all_chars(n_samples, **kwarg):
    symbols_list = []
    for alphabet in ALPHABET_MAP.values():
        symbols = alphabet.symbols[:200]
        logging.info("Using %d/%d symbols from alphabet %s", len(symbols), len(alphabet.symbols), alphabet.name)
        symbols_list.extend(zip(symbols, [alphabet] * len(symbols)))

    def attr_sampler():
        char, alphabet = symbols_list[np.random.choice(len(symbols_list))]
        return basic_image_sampler(alphabet=alphabet, char=char)()

    return dataset_generator(attr_sampler, n_samples)


def all_fonts(n_samples, **kwarg):
    font_list = []
    for alphabet in ALPHABET_MAP.values():
        fonts = alphabet.fonts[:500]
        logging.info("Using %d/%d fonts from alphabet %s", len(fonts), len(alphabet.fonts), alphabet.name)

        font_list.extend(zip(fonts, [alphabet] * len(fonts)))

    def attr_sampler():
        font, alphabet = font_list[np.random.choice(len(font_list))]
        return basic_image_sampler(alphabet=alphabet, font=font, is_bold=False, is_slant=False)()

    return dataset_generator(attr_sampler, n_samples)


def missing_symbol_dataset(n_samples, alphabet='latin', **kwarg):
    bg = MultiGradient(alpha=0.5, n_gradients=2, types=('linear', 'radial'))

    def tr(rng):
        if rng.rand() > 0.1:
            return tuple(rng.rand(2) * 2 - 1)
        else:
            return 10

    attr_generator = basic_image_sampler(alphabet=ALPHABET_MAP[alphabet], translation=tr, background=bg)
    return dataset_generator(attr_generator, n_samples)


def less_variations(n_samples, alphabet='latin', **kwarg):
    attr_generator = basic_image_sampler(
        alphabet=ALPHABET_MAP[alphabet], is_bold=False, is_slant=False,
        scale=lambda rng: 0.5 * np.exp(rng.randn() * 0.1),
        rotation=lambda rng: rng.randn() * 0.1)
    return dataset_generator(attr_generator, n_samples)


DATASET_GENERATOR_MAP = {
    'plain': generate_plain_dataset,
    'default': generate_default_dataset,
    'camouflage': generate_camouflage_dataset,
    'segmentation': generate_segmentation_dataset,
    'counting': generate_couting_dataset,
    'missing-symbol': missing_symbol_dataset,
    'tiny': generate_tiny_dataset,
    'all_fonts': all_fonts,
    'all_chars': all_chars,
    'less_variations': less_variations,
}
