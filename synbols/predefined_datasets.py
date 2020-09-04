import logging
import numpy as np

from .drawing import Camouflage, NoPattern, SolidColor, MultiGradient
from .fonts import ALPHABET_MAP
from .generate import dataset_generator, basic_attribute_sampler, flatten_mask, flatten_mask_except_first, add_occlusion


def generate_plain_dataset(n_samples, alphabet='latin', seed=None, **kwargs):
    """Generate white on black, centered symbols. The only factors of variations are font and char."""
    alphabet = ALPHABET_MAP[alphabet]
    attr_sampler = basic_attribute_sampler(
        alphabet=alphabet, background=NoPattern(), foreground=SolidColor((1, 1, 1,)), is_slant=False,
        is_bold=False, rotation=0, scale=1., translation=(0., 0.), inverse_color=False, pixel_noise_scale=0.)
    return dataset_generator(attr_sampler, n_samples, seed=seed)


def generate_tiny_dataset(n_samples, alphabet='latin', seed=None, **kwarg):
    """Generate a dataset of 8x8 resolution in gray scale with scale of 1 and minimal variations."""
    fg = SolidColor((1, 1, 1))
    bg = SolidColor((0, 0, 0))
    attr_sampler = basic_attribute_sampler(alphabet=ALPHABET_MAP[alphabet], background=bg, foreground=fg, is_bold=False,
                                           is_slant=False, scale=1, resolution=(8, 8), is_gray=True)
    return dataset_generator(attr_sampler, n_samples, seed=seed)


def generate_default_dataset(n_samples, alphabet='latin', seed=None, **kwarg):
    """Generate the default dataset, using gradiant as foreground and background."""
    attr_sampler = basic_attribute_sampler(alphabet=ALPHABET_MAP[alphabet])
    return dataset_generator(attr_sampler, n_samples, seed=seed)


def generate_solid_bg_dataset(n_samples, alphabet='latin', seed=None, **kwarg):
    """Same as default datasets, but uses white on black."""
    fg = SolidColor((1, 1, 1))
    bg = SolidColor((0, 0, 0))

    attr_sampler = basic_attribute_sampler(alphabet=ALPHABET_MAP[alphabet], background=bg, foreground=fg)
    return dataset_generator(attr_sampler, n_samples, seed=seed)


def generate_korean_1k_dataset(n_samples, seed=None, **kwarg):
    """Uses the first 1000 korean symbols"""
    chars = ALPHABET_MAP['korean'].symbols[:1000]
    fonts = ALPHABET_MAP['korean'].fonts
    attr_sampler = basic_attribute_sampler(char=lambda rng: rng.choice(chars), font=lambda rng: rng.choice(fonts))
    return dataset_generator(attr_sampler, n_samples, seed=seed)


def generate_camouflage_dataset(n_samples, alphabet='latin', texture='camouflage', seed=None, **kwarg):
    """Generate a dataset where the pixel distribution is the same for the foreground and background."""

    def attr_sampler(seed=None):
        if texture == 'camouflage':
            angle = 0
            fg = Camouflage(stroke_angle=angle, stroke_width=0.1, stroke_length=0.6, stroke_noise=0)
            bg = Camouflage(stroke_angle=angle + np.pi / 2, stroke_width=0.1, stroke_length=0.6, stroke_noise=0)
        elif texture == 'shade':
            fg, bg = None, None
        elif texture == 'bw':
            fg = SolidColor((1, 1, 1))
            bg = SolidColor((0, 0, 0))
        else:
            raise ValueError("Unknown texture %s." % texture)

        scale = 0.7 * np.exp(np.random.randn() * 0.1)
        return basic_attribute_sampler(
            alphabet=ALPHABET_MAP[alphabet], background=bg, foreground=fg, is_bold=True, is_slant=False,
            scale=scale)(seed)

    return dataset_generator(attr_sampler, n_samples, seed=seed)


def generate_non_camou_bw_dataset(n_samples, alphabet='latin', seed=None, **kwargs):
    """Generate a black and white dataset with same attribute distribution as the camouflage dataset."""
    return generate_camouflage_dataset(n_samples, alphabet=alphabet, texture='bw', seed=seed, **kwargs)


def generate_non_camou_shade_dataset(n_samples, alphabet='latin', seed=None, **kwargs):
    """Generate a gradient foreground and background dataset with same attribute distribution as the camouflage dataset.
    """
    return generate_camouflage_dataset(n_samples, alphabet=alphabet, texture='shade', seed=seed, **kwargs)


# for segmentation, detection, counting
# -------------------------------------

def generate_segmentation_dataset(n_samples, alphabet='latin', resolution=(128, 128), seed=None, **kwarg):
    """Generate 3-10 symbols of various scale and rotation and translation (no bold)."""

    def scale(rng):
        return 0.1 * np.exp(rng.randn() * 0.4)

    def n_symbols(rng):
        return rng.choice(list(range(3, 10)))

    attr_generator = basic_attribute_sampler(alphabet=ALPHABET_MAP[alphabet], resolution=resolution, scale=scale,
                                             is_bold=False, n_symbols=n_symbols)
    return dataset_generator(attr_generator, n_samples, flatten_mask, seed=seed)


def generate_counting_dataset(n_samples, alphabet='latin', resolution=(128, 128), n_symbols=None, scale_variation=0.5,
                              seed=None, **kwarg):
    """Generate 3-10 symbols at various scale. Samples 'a' with prob 70% or a latin lowercase otherwise."""

    if n_symbols is None:
        def n_symbols(rng):
            return rng.choice(list(range(3, 10)))

    def scale(rng):
        return 0.1 * np.exp(rng.randn() * scale_variation)

    def char_sampler(rng):
        if rng.rand() < 0.3:
            return rng.choice(ALPHABET_MAP[alphabet].symbols)
        else:
            return 'a'

    attr_generator = basic_attribute_sampler(alphabet=ALPHABET_MAP[alphabet], char=char_sampler, resolution=resolution,
                                             scale=scale, is_bold=False, n_symbols=n_symbols)
    return dataset_generator(attr_generator, n_samples, flatten_mask, seed=seed)


def generate_counting_dataset_scale_fix(n_samples, seed=None, **kwargs):
    """Generate 3-10 symbols at fixed scale. Samples 'a' with prob 70% or a latin lowercase otherwise."""
    return generate_counting_dataset(n_samples, scale_variation=0, seed=seed, **kwargs)


def generate_counting_dataset_crowded(n_samples, seed=None, **kwargs):
    """Generate 30-50 symbols at fixed scale. Samples 'a' with prob 70% or a latin lowercase otherwise."""

    def n_symbols(rng):
        return rng.choice(list(range(30, 50)))

    return generate_counting_dataset(n_samples, scale_variation=0.1, n_symbols=n_symbols, seed=seed, **kwargs)


# for few-shot learning
# ---------------------

def all_chars(n_samples, seed=None, **kwarg):
    """Combines the symbols of all languages (up to 200 per languages). Note: some fonts may appear rarely."""

    symbols_list = []
    for alphabet in ALPHABET_MAP.values():
        symbols = alphabet.symbols[:200]
        logging.info("Using %d/%d symbols from alphabet %s", len(symbols), len(alphabet.symbols), alphabet.name)
        symbols_list.extend(zip(symbols, [alphabet] * len(symbols)))

    def attr_sampler(seed=None):
        char, alphabet = symbols_list[np.random.choice(len(symbols_list))]
        return basic_attribute_sampler(alphabet=alphabet, char=char)(seed)

    return dataset_generator(attr_sampler, n_samples, seed=seed)


def generate_balanced_font_chars_dataset(n_samples, seed=None, **kwarg):
    """Samples uniformly from all fonts (max 200 per alphabet) or uniformly from all symbols (max 200 per alphabet)
    with probability 50%.
    """
    font_list = []
    symbols_list = []

    for alphabet in ALPHABET_MAP.values():
        fonts = alphabet.fonts[:200]
        symbols = alphabet.symbols[:200]

        logging.info("Using %d/%d fonts from alphabet %s", len(fonts), len(alphabet.fonts), alphabet.name)
        font_list.extend(zip(fonts, [alphabet] * len(fonts)))

        logging.info("Using %d/%d symbols from alphabet %s", len(symbols), len(alphabet.symbols), alphabet.name)
        symbols_list.extend(zip(symbols, [alphabet] * len(symbols)))

    logging.info("Total n_fonts: %d, n_symbols: %d.", len(font_list), len(symbols_list))

    def attr_sampler(seed=None):
        if np.random.rand() > 0.5:
            font, alphabet = font_list[np.random.choice(len(font_list))]
            symbol = np.random.choice(alphabet.symbols[:200])
        else:
            symbol, alphabet = symbols_list[np.random.choice(len(symbols_list))]
            font = np.random.choice(alphabet.fonts[:200])
        return basic_attribute_sampler(char=symbol, font=font, is_bold=False, is_slant=False)(seed)

    return dataset_generator(attr_sampler, n_samples, seed=seed)


# for active learning
# -------------------

def generate_large_translation(n_samples, alphabet='latin', seed=None, **kwarg):
    """Synbols are translated beyond the border of the image to create a cropping effect. Scale is fixed to 0.5."""
    attr_sampler = basic_attribute_sampler(alphabet=ALPHABET_MAP[alphabet], scale=0.5,
                                           translation=lambda rng: tuple(rng.rand(2) * 4 - 2))
    return dataset_generator(attr_sampler, n_samples, seed=seed)


def missing_symbol_dataset(n_samples, alphabet='latin', seed=None, **kwarg):
    """With 10% probability, no symbols are drawn"""
    bg = MultiGradient(alpha=0.5, n_gradients=2, types=('linear', 'radial'))

    def tr(rng):
        if rng.rand() > 0.1:
            return tuple(rng.rand(2) * 2 - 1)
        else:
            return 10

    attr_generator = basic_attribute_sampler(alphabet=ALPHABET_MAP[alphabet], translation=tr, background=bg)
    return dataset_generator(attr_generator, n_samples, seed=seed)


def generate_some_large_occlusions(n_samples, alphabet='latin', seed=None, **kwarg):
    """With probability 20%, add a large occlusion over the existing symbol."""

    def n_occlusion(rng):
        if rng.rand() < 0.2:
            return 1
        else:
            return 0

    attr_sampler = add_occlusion(basic_attribute_sampler(alphabet=ALPHABET_MAP[alphabet]),
                                 n_occlusion=n_occlusion,
                                 scale=lambda rng: 0.6 * np.exp(rng.randn() * 0.1),
                                 translation=lambda rng: tuple(rng.rand(2) * 6 - 3))
    return dataset_generator(attr_sampler, n_samples, flatten_mask_except_first, seed=seed)


def generate_many_small_occlusions(n_samples, alphabet='latin', seed=None, **kwarg):
    """Add small occlusions on all images. Number of occlusions are sampled uniformly in [0,5)."""
    attr_sampler = add_occlusion(basic_attribute_sampler(alphabet=ALPHABET_MAP[alphabet]),
                                 n_occlusion=lambda rng: rng.randint(0, 5))
    return dataset_generator(attr_sampler, n_samples, flatten_mask_except_first, seed=seed)


def generate_pixel_noise(n_samples, alphabet='latin', seed=None, **kwarg):
    """Add large pixel noise with probability 0.5."""

    def pixel_noise(rng):
        if rng.rand() > 0.5:
            return 0
        else:
            return 0.35

    attr_sampler = basic_attribute_sampler(alphabet=ALPHABET_MAP[alphabet], pixel_noise_scale=pixel_noise)
    return dataset_generator(attr_sampler, n_samples, seed=seed)


# for font classification
# -----------------------

def less_variations(n_samples, alphabet='latin', seed=None, **kwarg):
    """Less variations in scale and rotations. Also, no bold and no italic. This makes a more accessible font
    classification task."""
    attr_generator = basic_attribute_sampler(
        alphabet=ALPHABET_MAP[alphabet], is_bold=False, is_slant=False,
        scale=lambda rng: 0.5 * np.exp(rng.randn() * 0.1),
        rotation=lambda rng: rng.randn() * 0.1)
    return dataset_generator(attr_generator, n_samples, seed=seed)


DATASET_GENERATOR_MAP = {
    'plain': generate_plain_dataset,
    'default': generate_default_dataset,
    'default-bw': generate_solid_bg_dataset,
    'korean-1k': generate_korean_1k_dataset,
    'camouflage': generate_camouflage_dataset,
    'non-camou-bw': generate_non_camou_bw_dataset,
    'non-camou-shade': generate_non_camou_shade_dataset,
    'segmentation': generate_segmentation_dataset,
    'counting': generate_counting_dataset,
    'counting-fix-scale': generate_counting_dataset_scale_fix,
    'counting-crowded': generate_counting_dataset_crowded,
    'missing-symbol': missing_symbol_dataset,
    'some-large-occlusion': generate_some_large_occlusions,
    'many-small-occlusion': generate_many_small_occlusions,
    'large-translation': generate_large_translation,
    'tiny': generate_tiny_dataset,
    'balanced-font-chars': generate_balanced_font_chars_dataset,
    'all-chars': all_chars,
    'less-variations': less_variations,
    'pixel-noise': generate_pixel_noise,
}
