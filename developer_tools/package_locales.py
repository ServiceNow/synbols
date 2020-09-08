"""
Creates data files for each locale with supported characters and fonts

"""
import numpy as np
import os
import pickle

from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
from icu import Locale, LocaleData, ULocaleDataExemplarSetType, USET_ADD_CASE_MAPPINGS
from itertools import chain
from scipy.sparse import csr_matrix, lil_matrix, load_npz, save_npz
from subprocess import call, check_output
from synbols.drawing import Image, SolidColor, Symbol
from tqdm import tqdm
from urllib.request import urlretrieve


UNICODE_STANDARD_URL = 'https://www.unicode.org/Public/UNIDATA/Blocks.txt'


# Character type constants for the ICU library
# XXX: we currently exclude punctuation symbols because it's roughly the same for all languages
#      and including them dramatically increases the number of supported fonts for each locale
#      which slows down the testing of font properties.
ICU_CHAR_TYPES = {
                  "standard": ULocaleDataExemplarSetType.ES_STANDARD,
                  "auxiliary": ULocaleDataExemplarSetType.ES_AUXILIARY,
                  # "punctuation": 3
                  }


def get_locale_chars(locale, unicode=False):
    """
    Based on ICU examplars

    unicode=True returns the decimal code for the chars, otherwise the actual char.

    """
    _filter_valid = lambda chars: [x for x in chars if len(x) == 1 and "zero width" not in Unicode[ord(x)].lower()]

    all_chars = {}
    for char_type, char_type_id in ICU_CHAR_TYPES.items():
        chars = np.array(_filter_valid(list(LocaleData(locale).getExemplarSet(USET_ADD_CASE_MAPPINGS, char_type_id))))

        # Segment upper and lower if needed
        if char_type in ["standard", "auxiliary"]:
            if len(chars) > 0:
                # is lower if .lower() gives same char code
                is_lower = np.array([len(c) == len(c.lower()) and ord(c) == ord(c.lower()) for c in chars])
                all_chars[char_type + "_lower"] = chars[is_lower]
                all_chars[char_type + "_upper"] = chars[~is_lower]
            else:
                all_chars[char_type + "_lower"] = chars
                all_chars[char_type + "_upper"] = chars
        else:
            all_chars[char_type] = chars
        
    # Convert to codes if needed
    if unicode:
        all_chars = {k: [ord(x) for x in v] for k, v in all_chars.items()}
        
    return {k: v for k, v in all_chars.items()}


def get_sys_fonts():
    """
    Returns a list of all installed system fonts

    """
    cmd = ['fc-list']
    lines = check_output(cmd).splitlines()

    font_dict = {}
    for font_string in lines:
        font_path, font_string = str(font_string).split(':')[0:2]
        font_path = font_path[2:].strip()
        font_name = (font_string.split(',')[0].split("\\")[0]).strip()
        font_dict[font_name] = font_path

    return font_dict


def get_unicode_tables_by_font(font):
    """
    Returns the unicode codes of all characters supported by a font

    """
    ttf = TTFont(font, 0, allowVID=0,
                 ignoreDecompileErrors=True, fontNumber=-1)
    supported_chars = set(
        [key for table in ttf["cmap"].tables for key in table.cmap.keys()])
    ttf.close()
    return sorted(supported_chars)


def load_unicode_blocks():
    """
    Loads all unicode blocks into a dict indexed by name with start/stop indices as value.

    """
    _path = 'unicode_blocks.txt'
    urlretrieve(UNICODE_STANDARD_URL, _path)
    lines = [l.strip() for l in open(_path, 'r', encoding="utf-8")
             if len(l.strip()) > 0 and '#' not in l]
    lines = {l.split(';')[1].strip().lower(): {'start': int(l.split(';')[0].split('..')[0], base=16),
                                               'stop': int(l.split(';')[0].split('..')[1], base=16)}
             for l in lines}
    os.remove(_path)
    return lines


def test_font_properties(char_codes, fonts, glyph_avail):
    """
    Test visual font properties (e.g., bold, empty images)

    """
    def make_test_symbol(char, font, is_slant=False, is_bold=False):
        symbol = Symbol(alphabet="foo", char=char, font=font, foreground=SolidColor((1, 1, 1,)), 
                        is_slant=is_slant, is_bold=is_bold, rotation=0, scale=1, translation=(0, 0), 
                        rng=np.random.RandomState(42))
        return Image([symbol], background=SolidColor((0, 0, 0,)), inverse_color=False, resolution=(16, 16),
                     pixel_noise_scale=0).make_image()

    bold_works = np.zeros(len(fonts))
    render_works = np.zeros(len(fonts))

    n_true = 0
    for i, font in enumerate(fonts):
        supported_char_idx = np.where(glyph_avail[:, i] == 1)[0]
        if len(supported_char_idx) < 1:
            bold_works[i] = 0
            render_works[i] = 0
            continue
        
        codes = char_codes[np.random.choice(supported_char_idx, min(100, len(supported_char_idx)), replace=False)]
        
        bold_success = 0
        render_success = 0
        for code in codes:
            try:
                normal = make_test_symbol(chr(code), font, is_bold=False)
                bold = make_test_symbol(chr(code), font, is_bold=True)
                bold_success += 1 if (np.abs(normal - bold).sum() != 0) else 0
                render_success += 1 if (normal.sum() > 0) else 0
            except ZeroDivisionError:
                pass

        # We don't take the risk of including partially successful fonts
        bold_works[i] = bold_success == len(codes)
        render_works[i] = render_success == len(codes)

    return bold_works, render_works


if __name__ == "__main__":
    unicode_blocks = load_unicode_blocks()
    fonts = get_sys_fonts()
    font_names = np.array(list(fonts.keys()))
    import sys
    print("Found %d system fonts" % len(fonts), file=sys.stderr)

    # Make a huge sparse binary matrix that gives the availability of glyphs for each char in each font
    glyph_avail = lil_matrix((max(b['stop'] for b in unicode_blocks.values()), len(font_names)), dtype=np.uint8)
    for i, (name, font) in tqdm(enumerate(fonts.items()), total=len(fonts), desc="Checking glyph availability"):
        chars = get_unicode_tables_by_font(font)
        glyph_avail[chars, i] = 1

    # Package all locales
    locales = [(k, v) for k, v in Locale.getAvailableLocales().items()
               if "_" not in k and
               k in ['en', 'te', 'th', 'vi', 'ar', 'iw', 'km', 'ta', 'gu', 'bn', 'ml', 'el', 'ru', 'ko', 'zh', 'jp']]
    for code, locale in tqdm(locales, desc="Packaging locales"):
        chars = get_locale_chars(code, unicode=True)
        name = locale.getDisplayName().encode('ascii', 'ignore').decode('ascii')

        char_codes = np.sort(list(chain(*chars.values())))  # Unicode code of each char
        row_by_code = dict(zip(char_codes, range(len(char_codes))))  # Where each code appears in the resulting matrix

        # Convert the char dict to use matrix indices
        # The resulting dict allows to filter the locale's matrix directly
        char_types = {k: [row_by_code[x] for x in v] for k, v in chars.items()}
        locale_glyph_avail = np.asarray(glyph_avail[char_codes].todense())

        # # Keep only fonts that support at least one character
        # TODO: doesnt really work with punctuation so I removed it for now
        # mask = locale_glyph_avail.sum(axis=0).reshape(-1,) > 0.1 * locale_glyph_avail.shape[0]
        # assert mask.shape[0] == len(fonts)
        # locale_glyph_avail = locale_glyph_avail[:, mask]
        # locale_font_idx = np.where(mask)[0]

        locale_bold_avail, locale_render_works = test_font_properties(char_codes, font_names, locale_glyph_avail)
        
        # Filter fonts that don't render properly
        mask = np.logical_and(locale_render_works, locale_glyph_avail.sum(axis=0) > 1)
        locale_font_idx = np.where(mask)[0]
        locale_bold_avail = locale_bold_avail[mask]
        locale_glyph_avail = locale_glyph_avail[:, mask]

        # XXX: We could make this use less disk space by storing the index of fonts instead of the names
        #      but for now we will store full names for easy introspection of locale files.
        data_bundle = dict(glyph_avail=locale_glyph_avail,
                           bold_avail=locale_bold_avail,
                           char_codes=char_codes,
                           fonts=font_names[locale_font_idx])
        data_bundle.update({"char_types__" + k: v for k, v in char_types.items()})
        np.savez_compressed("locale_%s_%s.npz" % (code, name.lower()), **data_bundle)
