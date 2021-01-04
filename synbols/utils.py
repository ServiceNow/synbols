import csv
import os
from collections import defaultdict, Counter
from itertools import chain
from warnings import warn

import numpy as np

LOCALE_DATA_PATH = "/locales"


class Alphabet:
    """Combines fonts and symbols for a given language."""

    def __init__(self, name, fonts, symbols):
        self.name = name
        self.symbols = symbols
        self.fonts = fonts


def flatten_attr(attr, ctxt=None):
    flat_dict = {}
    if isinstance(attr, (list, tuple, np.ndarray)):
        for i, val in enumerate(attr):
            flat_dict.update(flatten_attr(val, ctxt + '[%d]' % i))

    elif isinstance(attr, dict):
        for key, val in attr.items():
            if ctxt is None:
                sub_ctxt = key
            else:
                sub_ctxt = ctxt + '.%s' % key
            flat_dict.update(flatten_attr(val, sub_ctxt))
    else:
        flat_dict = {ctxt: attr}
    return flat_dict


def _extract_axis(y, axis_name, max_val):
    if axis_name is None:
        return [None] * max_val

    counter = Counter([attr[axis_name] for attr in y])
    return [e for e, _ in counter.most_common(max_val)]


def make_img_grid(x, y, h_axis='char', v_axis='font', n_row=20, n_col=40):
    h_values = _extract_axis(y, h_axis, n_col)
    v_values = _extract_axis(y, v_axis, n_row)

    attr_map = defaultdict(list)
    for i, attr in enumerate(y):
        attr_map[(attr.get(h_axis), attr.get(v_axis))].append(i)

    img_grid = []
    blank_image = np.zeros(x.shape[1:], dtype=x.dtype)

    for v_value in v_values:
        img_row = []
        for h_value in h_values:
            if len(attr_map[(h_value, v_value)]) > 0:

                idx = attr_map[(h_value, v_value)].pop(0)
                img_row.append(x[idx])
            else:
                img_row.append(blank_image)

        img_grid.append(np.hstack(img_row))

    img_grid = np.vstack(img_grid)

    if img_grid.shape[-1] == 1:
        img_grid = img_grid.squeeze(axis=-1)

    return img_grid, h_values, v_values


class FailSafeLanguage:
    def get_alphabet(self, standard=True, auxiliary=False, lower=True, upper=False, support_bold=False,
                     include_blacklisted_fonts=False):

        chars = list('abcdefghijklmnopqrstuvwxyz')
        return  Alphabet(name='english', symbols=chars, fonts=['arial'])


class Language:
    def __init__(self, locale_file, font_blacklist_dir):
        self.data_file = locale_file
        self.font_blacklist_dir = font_blacklist_dir
        try:
            self.name = os.path.basename(self.data_file) \
                .replace(".npz", "").replace("locale_", "").split("_")[1].lower()
        except Exception:
            print(locale_file)

        self.loaded = False

    def _load_data(self):
        # Load locale data
        data = np.load(self.data_file)
        self.char_types = {k.replace("char_types__", ""): v for k, v in data.items() if "char_types__" in k}
        self.char_codes = data["char_codes"].astype(np.uint)
        self.glyph_avail = data["glyph_avail"]
        self.fonts = data["fonts"]
        self.bold_avail = data["bold_avail"].astype(np.bool)
        del data

        blacklist_file = os.path.join(self.font_blacklist_dir, 'blacklist_%s.tsv' % self.name)
        if os.path.exists(blacklist_file):
            self.font_blacklist = _read_blacklist_file(blacklist_file)
        else:
            self.font_blacklist = []

        self.loaded = True

    def get_alphabet(self,
                     standard=True,
                     auxiliary=True,
                     lower=True,
                     upper=False,
                     support_bold=False,
                     include_blacklisted_fonts=False):
        # Load locale data on demand
        if not self.loaded:
            self._load_data()

        # Assemble character indices
        chars_to_keep = []
        if standard:
            if lower:
                chars_to_keep.append(self.char_types["standard_lower"])
            if upper:
                chars_to_keep.append(self.char_types["standard_upper"])
        if auxiliary:
            if lower:
                chars_to_keep.append(self.char_types["auxiliary_lower"])
            if upper:
                chars_to_keep.append(self.char_types["auxiliary_upper"])

        # Validate final selection
        chars_to_keep = list(chain(*chars_to_keep))
        if len(chars_to_keep) == 0:
            raise ValueError("Filtered character set is empty. \
            Consider including more characters using the arguments.")
        chars_to_keep = np.array(chars_to_keep)

        char_codes = self.char_codes[chars_to_keep]
        glyph_avail = self.glyph_avail[chars_to_keep]

        # Filter fonts based on boldness
        if support_bold:
            fonts = self.fonts[self.bold_avail]
            glyph_avail = glyph_avail[:, self.bold_avail]
        else:
            fonts = self.fonts

        # Extract chunk using heuristic
        # -- Heuristic beings
        char_support = glyph_avail.sum(axis=0) / glyph_avail.shape[0]
        min_support = 0.8 * max(char_support)

        # Keep only fonts that have the minimum support
        mask = char_support >= min_support
        assert mask.shape[0] == glyph_avail.shape[1]
        glyph_avail = glyph_avail[:, mask]
        fonts = fonts[mask]

        # Drop all chars not supported by the remaining fonts
        mask = glyph_avail.sum(axis=1) == glyph_avail.shape[1]
        assert mask.shape[0] == glyph_avail.shape[0]
        char_codes = char_codes[mask]
        glyph_avail = glyph_avail[mask]
        # -- Heuristic ends

        if not include_blacklisted_fonts:
            # print("blacklisting the following \n%s"% ('\n'.join(self.font_blacklist)))
            fonts = np.setdiff1d(fonts, self.font_blacklist)

        # Return chars and fonts
        return Alphabet(self.name,
                        fonts=fonts,
                        symbols=[chr(x) for x in char_codes])


def language_map_statistics():
    str_list = []
    lang_map = load_all_languages()
    for lang_name, lang in lang_map.items():
        alphabet = lang.get_alphabet()
        str_list.append("  * Language %s contains %d fonts and %d symbols" % (
            lang_name, len(alphabet.fonts), len(alphabet.symbols)))
    return "\n".join(str_list)


def _read_blacklist_file(file_path):
    with open(file_path, 'r') as fd:
        blacklist = [row[0] for row in csv.reader(fd, delimiter="\t")]
    return blacklist


def load_all_languages(override_locale_path=None):
    """
    Loads all supported languages.
    Returns a dictionnary of Language objects indexed by their name.

    """
    locale_path = LOCALE_DATA_PATH \
        if override_locale_path is None \
        else override_locale_path
    blacklist_dir = os.path.join(os.path.dirname(__file__), 'fonts', 'blacklist')

    languages = {}
    if os.path.exists(locale_path):
        for locale_file in [os.path.join(locale_path, x)
                            for x in os.listdir(locale_path)
                            if x.startswith("locale_") and x.endswith(".npz")]:
            lang = Language(locale_file=locale_file, font_blacklist_dir=blacklist_dir)
            languages[lang.name] = lang
    else:
        warn("The locale data path was not found. \
        Did you execute the code with the 'synbols' executable?")
        languages['english'] = FailSafeLanguage()
    return languages
