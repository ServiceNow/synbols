import numpy as np
import logging

from ..drawing import Image, SolidColor, Symbol


def make_test_symbol(char, font, is_slant=False, is_bold=False):
    symbol = Symbol(alphabet="foo", char=char, font=font, foreground=SolidColor((1, 1, 1,)), is_slant=is_slant,
                    is_bold=is_bold, rotation=0, scale=1, translation=(0, 0), rng=np.random.RandomState(42))
    return Image([symbol], background=SolidColor((0, 0, 0,)), inverse_color=False, resolution=(16, 16),
                 pixel_noise_scale=0).make_image()


def check_errors(font, alphabet, max_chars=None):
    """Checks for empty canvas, slant or bold issues."""
    empty_count = 0
    no_bold_count = 0
    no_slant_count = 0
    no_case_count = 0

    char_list = alphabet.symbols
    if max_chars is not None:
        char_list = np.random.choice(char_list, min(max_chars, len(char_list)), replace=False)

    for char in char_list:
        plain = make_test_symbol(font=font, char=char)
        bold = make_test_symbol(font=font, char=char, is_bold=True)
        # slant = make_test_symbol(font=font, char=char, is_slant=True)

        if plain.sum() == 0:
            empty_count += 1

        if np.abs(plain - bold).sum() == 0:
            no_bold_count += 1

        # if np.abs(plain - slant).sum() == 0:
        #     no_slant_count += 1

        if char.swapcase() != char:
            swap_case = make_test_symbol(font=font, char=char.swapcase())
            if np.abs(plain - swap_case).sum() == 0:
                no_case_count += 1

    all_good = (empty_count + no_bold_count + no_slant_count + no_case_count) == 0

    return dict(empty_count=empty_count,
                no_bold_count=no_bold_count,
                no_slant_count=no_slant_count,
                no_case_count=no_case_count,
                all_good=all_good)


def eval_all_font_properties(alphabet_map, max_chars=None):
    """
    Runs a bunch of checks on all fonts for all alphabets
    """

    property_map = {}

    for i, alphabet in enumerate(alphabet_map.values()):
        logging.info("%2d/%d alphabet %s, %d fonts, %d symbols",
                     i + 1, len(alphabet_map), alphabet.name, len(alphabet.fonts), len(alphabet.symbols))
        # whitelist[alphabet.name] = filter_fonts(alphabet)[0]

        property_map[alphabet.name] = {}

        good_count = 0
        for i, font in enumerate(alphabet.all_fonts):
            if max_chars is not None:
                n_chars = min(max_chars, len(alphabet.symbols))
            else:
                n_chars = alphabet.symbols

            logging.debug("Checking font %s for alphabet %s (%d/%d)", font, alphabet.name, i + 1, len(alphabet.fonts))
            font_properties = check_errors(font, alphabet, max_chars=max_chars)
            property_map[alphabet.name][font] = font_properties

            if font_properties['all_good']:
                good_count += 1
                logging.debug("    Supports all characters")
            else:
                logging.info("Errors for font %s of alphabet %s.", font, alphabet.name)
                for key, val in font_properties.items():
                    if (val > 0) and (key != 'all_good'):
                        logging.info("    %s: %d / %d", key, val, n_chars)

        logging.info("%d / %d fonts pass all tests", good_count, len(alphabet.fonts))
        logging.info("============================")

    return property_map
