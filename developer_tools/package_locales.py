import json
import numpy as np
import pickle

from fontTools.unicode import Unicode
from icu import Locale, LocaleData, ULocaleDataExemplarSetType, USET_ADD_CASE_MAPPINGS
from itertools import chain
from scipy.sparse import load_npz
from synbols.drawing import Image, SolidColor, Symbol


ICU_CHAR_TYPES = {
                  "standard": ULocaleDataExemplarSetType.ES_STANDARD,
                  "auxiliary": ULocaleDataExemplarSetType.ES_AUXILIARY,
                #   "punctuation": 3  # TODO: much more efficient when we remove punctuation since most fonts support it and its common to most languages
                  }


def get_chars(locale, unicode=False):
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
                # is upper if .upper() gives same char code
                is_upper = np.array([len(c) == len(c.upper()) and ord(c) == ord(c.upper()) for c in chars])
                all_chars[char_type + "_lower"] = chars[~is_upper]
                all_chars[char_type + "_upper"] = chars[is_upper]
            else:
                all_chars[char_type + "_lower"] = chars
                all_chars[char_type + "_upper"] = chars
        else:
            all_chars[char_type] = chars
        
    # Convert to codes if needed
    if unicode:
        all_chars = {k: [ord(x) for x in v] for k, v in all_chars.items()}
        
    return {k: v for k, v in all_chars.items()}


def make_test_symbol(char, font, is_slant=False, is_bold=False):
    symbol = Symbol(alphabet="foo", char=char, font=font, foreground=SolidColor((1, 1, 1,)), is_slant=is_slant,
                    is_bold=is_bold, rotation=0, scale=1, translation=(0, 0), rng=np.random.RandomState(42))
    return Image([symbol], background=SolidColor((0, 0, 0,)), inverse_color=False, resolution=(16, 16),
                 pixel_noise_scale=0).make_image()


def bold_support_matrix(char_codes, fonts, glyph_avail):
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
    fonts = pickle.load(open("./developer_tools/fonts.pkl", "rb"))
    font_names = np.array(list(fonts.keys()))
    glyph_avail = load_npz('./developer_tools/font_glyph_availability.npz')

    json.dump(font_names.tolist(), open("locale_font_names.json", "w"))

    for code, locale in [(k, v) for k, v in Locale.getAvailableLocales().items() if "_" not in k and k in ["en", "fr", "el", "te", "km", "ru", "hi", "vi", "ko", "ja"]]:
        chars = get_chars(code, unicode=True)
        name = locale.getDisplayName().encode('ascii', 'ignore').decode('ascii')
        print(name)

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

        locale_bold_avail, locale_render_works = bold_support_matrix(char_codes, font_names, locale_glyph_avail)
        
        # Filter fonts that don't render properly
        mask = np.logical_and(locale_render_works, locale_glyph_avail.sum(axis=0) > 1)
        locale_font_idx = np.where(mask)[0]
        locale_bold_avail = locale_bold_avail[mask]
        locale_glyph_avail = locale_glyph_avail[:, mask]

        np.savez_compressed("locale_%s.npz" % code, 
                            glyph_avail=locale_glyph_avail,
                            font_idx=locale_font_idx,
                            bold_avail=locale_bold_avail,
                            char_codes=char_codes)

        metadata = {
                    "char_types": char_types,
                    "name": name
                    }
        json.dump(metadata, open("locale_%s_metadata.json" % code, "w"))
