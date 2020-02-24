#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cairo
import numpy as np
# from osx_non_free_fonts import LATIN_FONTS, FUNKY_LATIN_FONTS, GREEK_FONTS, CHINESE_FONTS, CYRILLIC_FONTS, CHINESE_SYMBOLS
from google_fonts import LATIN_FONTS, GREEK_FONTS

class Language:
    """Combines fonts and symbols for a given language."""

    def __init__(self, fonts, symbols):
        self.symbols = symbols
        self.fonts = fonts

    def partition(self, ratio_dict, rng=np.random.RandomState(42)):
        """Splits both fonts and symbols into different partitions to define meta-train, meta-valid and meta-test.

        Args:
            ratio_dict: dict mapping the name of the subset to the ratio it should contain
            rng: a random number generator defining the randomness of the split

        Returns:
            A dict mapping the name of the subset to a new Language instance.
        """
        set_names, ratios = zip(*ratio_dict.items())
        symbol_splits = _split(self.symbols, ratios, rng)
        font_splits = _split(self.fonts, ratios, rng)
        return {set_name: Language(fonts, symbols)
                for set_name, symbols, fonts in zip(set_names, symbol_splits, font_splits)}


LANGUAGES = {
    'latin': Language(
        LATIN_FONTS,
        list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")),
    # 'latin': Language(
    #     LATIN_FONTS + FUNKY_LATIN_FONTS,
    #     list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")),
    # 'greek': Language(
    #     GREEK_FONTS,
    #     list(u"ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω")),
    # 'cyrillic': Language(
    #     CYRILLIC_FONTS,
    #     list(u"АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя")),
    # 'chinese': Language(CHINESE_FONTS, CHINESE_SYMBOLS),
    # # 'korean': Language(CHINESE_FONTS, KOREAN_SYLLABLES)
}


def write_char(ctxt, char, font_family, slant=cairo.FONT_SLANT_NORMAL, is_bold=False, rng=np.random):
    make_background(ctxt)

    weight = cairo.FONT_WEIGHT_BOLD if is_bold else cairo.FONT_WEIGHT_NORMAL

    ctxt.set_font_size(0.7)
    ctxt.select_font_face(font_family, slant, weight)
    extent = ctxt.text_extents(char)
    if len(char) == 3:
        extent_main_char = ctxt.text_extents(char[1])
    elif len(char) == 1:
        extent_main_char = extent
    else:
        raise Exception("Unexpected length of string: %d. Should be either 3 or 1" % len(char))

    if extent_main_char.width == 0. or extent_main_char.height == 0:
        print(char, font_family)
        return

    ctxt.translate(0.5, 0.5)
    scale = 0.6 / np.maximum(extent_main_char.width, extent_main_char.height)
    ctxt.scale(scale, scale)
    ctxt.scale(*np.exp(rng.randn(2) * 0.1))

    ctxt.rotate(rng.randn() * 0.2)

    if len(char) == 3:
        ctxt.translate(-ctxt.text_extents(char[0]).x_advance - extent_main_char.width / 2., extent_main_char.height / 2)
        ctxt.translate(*(rng.rand(2) * 0.1 - 0.05))
    else:
        ctxt.translate(-extent.width / 2., extent.height / 2)
        ctxt.translate(*(rng.rand(2) * 0.2 - 0.1))

    pat = random_pattern(0.8, (0.2, 1), patern_types=('linear',), rng=rng)
    ctxt.set_source(pat)
    ctxt.show_text(char)

    return extent


def make_char_np(width, height, char, font_family, slant, is_bold, rng=np.random):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctxt = cairo.Context(surface)
    ctxt.scale(width, height)  # Normalizing the canvas
    write_char(ctxt, char, font_family, slant, is_bold, rng)
    buf = surface.get_data()
    img = np.ndarray(shape=(width, height, 4), dtype=np.uint8, buffer=buf)
    img = img.astype(np.float32) / 256.
    img = img[:, :, 0:3]
    if rng.choice([True, False]):
        img = 1 - img

    min, max = np.min(img), np.max(img)
    img = (img - min) / (max - min)

    img += rng.randn(*img.shape) * 0.01
    img = np.clip(img, 0., 1.)

    return img


def random_pattern(alpha=0.8, brightness_range=(0, 1), patern_types=('linear', 'radial'), rng=np.random):
    pattern_type = rng.choice(patern_types)
    if pattern_type == 'linear':
        y1, y2 = rng.rand(2)
        pat = cairo.LinearGradient(-1, y1, 2, y2)
    if pattern_type == 'radial':
        x1, y1, x2, y2 = rng.randn(4) * 0.5
        pat = cairo.RadialGradient(x1 * 2, y1 * 2, 2, x2, y2, 0.2)

    def random_color():
        b_delta = brightness_range[1] - brightness_range[0]
        return rng.rand(3) * b_delta + brightness_range[0]

    r, g, b = random_color()
    pat.add_color_stop_rgba(1, r, g, b, alpha)
    r, g, b = random_color()
    pat.add_color_stop_rgba(0.5, r, g, b, alpha)
    r, g, b = random_color()
    pat.add_color_stop_rgba(0, r, g, b, alpha)
    return pat


def solid_pattern(alpha=0.8, brightness_range=(0, 1), rng=np.random):
    def random_color():
        b_delta = brightness_range[1] - brightness_range[0]
        return rng.rand(3) * b_delta + brightness_range[0]

    r, g, b = random_color()
    return cairo.SolidPattern(r, g, b, alpha)


def make_background(ctxt, rng=np.random):
    for i in range(5):
        pat = random_pattern(0.4, (0, 0.8), rng=rng)
        ctxt.rectangle(0, 0, 1, 1)  # Rectangle(x0, y0, x1, y1)
        ctxt.set_source(pat)
        ctxt.fill()


def _split(set_, ratios, rng=np.random.RandomState(42)):
    n = len(set_)
    counts = np.round(np.array(ratios) * n).astype(np.int)
    counts[0] = n - np.sum(counts[1:])
    set_ = rng.permutation(set_)
    idx = 0
    sets = []
    for count in counts:
        sets.append(set_[idx:(idx + count)])
        idx += count
    return sets


def make_ds_from_lang(lang, width, height, n_samples, rng=np.random.RandomState(42)):
    dataset = []
    for char in lang.symbols:
        one_class = []
        print(u"generating sample for char %s" % char)
        for i in range(n_samples):
            font = rng.choice(lang.fonts)
            x = make_char_np(width, height, char, font, rng.choice(3), rng.choice([True, False]), rng)
            one_class.append(x)

        dataset.append(np.stack(one_class))
    return dataset
