import cairo
import numpy as np

from functools import partial

from ..drawing import Attributes


StableAttribute = partial(Attributes, alphabet="foo", font="bar", background=None, foreground=None,
                          slant=cairo.FONT_SLANT_NORMAL, is_bold=False, rotation=0, scale=(1, 1),
                          translation=(0, 0), inverse_color=False, pixel_noise_scale=0.01, resolution=(32, 32), rng=42)


def check_empty_canvas(font, alphabet):
    """
    Checks if empty images are produced for a given alphabet and font with normal weight and slant

    """
    for char in alphabet.symbols:
        a = StableAttribute(alphabet=alphabet, font=font, char=char, is_bold=False,
                            slant=cairo.FONT_SLANT_NORMAL).make_image()
        if a.sum() == 0:
            return False
    return True


# TODO: maybe check for combinations that fail (e.g., bold/italic)
def check_rendering_bold(font, alphabet):
    """
    Checks if the appearance of characters changes from normal to bold face for a given alphabet and font

    """
    for char in alphabet.symbols:
        a1 = StableAttribute(alphabet=alphabet, font=font, char=char, is_bold=False).make_image()
        a2 = StableAttribute(alphabet=alphabet, font=font, char=char, is_bold=True).make_image()
        if np.abs(a1 - a2).sum() == 0:
            return False
    return True


def check_rendering_slant_italic(font, alphabet):
    """
    Checks if the appearance of characters changes when changing the slant from normal to italic for a given alphabet
    and font

    """
    for char in alphabet.symbols:
        a1 = StableAttribute(alphabet=alphabet, font=font, char=char, slant=cairo.FONT_SLANT_NORMAL).make_image()
        a2 = StableAttribute(alphabet=alphabet, font=font, char=char, slant=cairo.FONT_SLANT_ITALIC).make_image()
        if np.abs(a1 - a2).sum() == 0:
            return False
    return True


def check_rendering_slant_oblique(font, alphabet):
    """
    Checks if the appearance of characters changes when changing the slant from normal to oblique for a given alphabet
    and font

    """
    for char in alphabet.symbols:
        a1 = StableAttribute(alphabet=alphabet, font=font, char=char, slant=cairo.FONT_SLANT_NORMAL).make_image()
        a2 = StableAttribute(alphabet=alphabet, font=font, char=char, slant=cairo.FONT_SLANT_OBLIQUE).make_image()
        if np.abs(a1 - a2).sum() == 0:
            return False
    return True


def filter_fonts(alphabet):
    """
    Runs a bunch of checks on the fonts for an a

    """
    blacklist = set()

    def filter(test, font, alphabet):
        if test(font, alphabet):
            print("PASS")
        else:
            print("FAIL")
            blacklist.add(font)

    for font in alphabet.fonts:
        print("Checking font %s for alphabet %s" % (font, alphabet.name))

        print("--> Supports all characters.", end=" ")
        filter(check_empty_canvas, font, alphabet)

        print("--> Supports bold.", end=" ")
        filter(check_rendering_bold, font, alphabet)

        print("--> Supports italic.", end=" ")
        filter(check_rendering_slant_italic, font, alphabet)

        print("--> Supports oblique.", end=" ")
        filter(check_rendering_slant_oblique, font, alphabet)

    whitelist = set(alphabet.fonts).difference(blacklist)

    return list(whitelist), list(blacklist)