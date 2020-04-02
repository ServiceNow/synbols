import numpy as np
import logging

from ..drawing import Image, SolidColor, Symbol


def make_test_symbol(char, font, is_slant=False, is_bold=False):
    symbol = Symbol(alphabet="foo", char=char, font=font, foreground=SolidColor((1, 1, 1,)), is_slant=is_slant,
                    is_bold=is_bold, rotation=0, scale=(1, 1), translation=(0, 0), rng=np.random.RandomState(42))
    return Image([symbol], background=SolidColor((0, 0, 0,)), inverse_color=False, resolution=(16, 16),
                 pixel_noise_scale=0)


def check_errors(font, alphabet):
    """Checks for empty canvas, slant or bold issues."""
    empty_count = 0
    no_bold_count = 0
    no_slant_count = 0
    for char in alphabet.symbols:
        plain = make_test_symbol(font=font, char=char, is_bold=False).make_image()
        bold = make_test_symbol(font=font, char=char, is_bold=True).make_image()
        slant = make_test_symbol(font=font, char=char, is_slant=True).make_image()

        if plain.sum() == 0:
            empty_count += 1

        if np.abs(plain - bold).sum() == 0:
            no_bold_count += 1

        if np.abs(plain - slant).sum() == 0:
            no_slant_count += 1
    return empty_count, no_bold_count, no_slant_count


def filter_fonts(alphabet):
    """
    Runs a bunch of checks on the fonts for an a

    """
    blacklist = set()

    for i, font in enumerate(alphabet.fonts):
        logging.info("Checking font %s for alphabet %s (%d/%d)", font, alphabet.name, i + 1, len(alphabet.fonts))
        empty_count, no_bold_count, no_slant_count = check_errors(font, alphabet)
        if empty_count + no_bold_count + no_slant_count == 0:
            logging.info("    Supports all characters")
        else:
            logging.info("    Empty Count: %d", empty_count)
            logging.info("    No Bold Count: %d", no_bold_count)
            logging.info("    No Slant Count: %d", no_slant_count)

            # TODO: Somethimes glyphs fail to render and produce a square with a X inside. Would be good to detect that.

    whitelist = set(alphabet.fonts).difference(blacklist)

    return list(whitelist), list(blacklist)
