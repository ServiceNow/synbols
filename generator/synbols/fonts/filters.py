import numpy as np

from ..drawing import Image, SolidColor, Symbol


def make_test_symbol(char, font, is_slant=False, is_bold=False):
    symbol = Symbol(alphabet="foo", char=char, font=font, foreground=SolidColor((1, 1, 1,)), is_slant=is_slant,
                    is_bold=is_bold, rotation=0, scale=(1, 1), translation=(0, 0), rng=np.random.RandomState(42))
    return Image([symbol], background=SolidColor((0, 0, 0,)), inverse_color=False, resolution=(64, 64),
                 pixel_noise_scale=0.01)


def check_empty_canvas(font, alphabet):
    """
    Checks if empty images are produced for a given alphabet and font with normal weight and slant

    """
    for char in alphabet.symbols:
        a = make_test_symbol(font=font, char=char, is_bold=False).make_image()
        if a.sum() == 0:
            return False
    return True


def check_rendering_bold(font, alphabet):
    """
    Checks if the appearance of characters changes from normal to bold face for a given alphabet and font

    """
    for char in alphabet.symbols:
        a1 = make_test_symbol(font=font, char=char, is_bold=False).make_image()
        a2 = make_test_symbol(font=font, char=char, is_bold=True).make_image()
        if np.abs(a1 - a2).sum() == 0:
            return False
    return True


def check_rendering_slant(font, alphabet):
    """
    Checks if the appearance of characters changes when changing the slant for a given alphabet and font

    """
    for char in alphabet.symbols:
        a1 = make_test_symbol(font=font, char=char, is_slant=False).make_image()
        a2 = make_test_symbol(font=font, char=char, is_slant=True).make_image()
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
        # TODO: Somethimes glyphs fail to render and produce a square with a X inside. Would be good to detect that.
        filter(check_empty_canvas, font, alphabet)

        print("--> Supports bold.", end=" ")
        filter(check_rendering_bold, font, alphabet)

        print("--> Supports slant.", end=" ")
        filter(check_rendering_slant, font, alphabet)

    whitelist = set(alphabet.fonts).difference(blacklist)

    return list(whitelist), list(blacklist)