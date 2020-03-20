#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from icu import LocaleData
from os.path import join, exists
import logging

from font_utils import check_font

FONT_PATH = "/usr/share/fonts/truetype/google-fonts/"
METADATA = join(FONT_PATH, "google_fonts_metadata")
FONT_BLACKLIST = ["rubik", "podkova", "baloochettan2"]  # Fonts with known rendering issues

# Number of fonts per alphabet
# ----------------------------
# latin : 1011 (done)
# latin-ext : 701 (TODO: not sure we need extended yet)
# telugu : 22 (done)
# thai : 26 (done)
# vietnamese : 210 (done) (mostly latin with accents not sure if there is a big value)
# devanagari : 49 (TODO: cant really find a locale for it. Need to get chars from unicode)
# korean : 24 (done)
# arabic : 20 (done)
# cyrillic : 115 (done)
# cyrillic-ext : 89 (TODO: not sure we need extended yet)
# greek : 48 (done)
# greek-ext : 36 (TODO: not sure we need extended yet)
# hebrew : 17  (done)
# khmer : 24 (TODO: chars of length 2, what to do?)
# tamil : 14 (done)
# chinese-simplified : 7 (done)
# gujarati : 9 (done)
# bengali : 7 (done)
# malayalam : 6 (done)


# Less than 5
# ------------
# sinhala : 5
# tibetan : 2
# myanmar : 2
# oriya : 2
# lao : 2
# gurmukhi : 4
# ethiopic : 1
# japanese : 1
# kannada : 3

SYMBOL_MAP = {
    'latin': list(LocaleData("en_US").getExemplarSet()),
    'telugu': list(LocaleData("te").getExemplarSet()),
    'thai': list(LocaleData("th").getExemplarSet()),
    'vietnamese': list(LocaleData("vi").getExemplarSet()),
    'arabic': list(LocaleData("ar").getExemplarSet()),
    'hebrew': list(LocaleData("iw_IL").getExemplarSet()),
    # 'khmer': list(LocaleData("km").getExemplarSet()),  # XXX: see note above
    'tamil': list(LocaleData("ta").getExemplarSet()),
    'gujarati': list(LocaleData("gu").getExemplarSet()),
    'bengali': list(LocaleData("bn").getExemplarSet()),
    'malayalam': list(LocaleData("ml").getExemplarSet()),
    'greek': list(LocaleData("el_GR").getExemplarSet()),
    'cyrillic': list(u"АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"),
    'korean': list(LocaleData("ko_KR").getExemplarSet()),
    'chinese-simplified': list(LocaleData("zh-CN").getExemplarSet())
}


def parse_metadata(file_path):
    alphabet_map = defaultdict(list)
    font_map = defaultdict(list)

    for line in open(file_path, 'r'):
        elements = line.split(',')
        font_name = elements[0].strip()
        for alphabet in elements[1:]:
            alphabet = alphabet.strip()
            alphabet_map[alphabet].append(font_name)
            font_map[font_name].append(alphabet)

    return alphabet_map, font_map


class Alphabet:
    """Combines fonts and symbols for a given language."""

    def __init__(self, name, fonts, symbols):
        self.name = name
        self.symbols = symbols
        self.fonts = fonts


def build_alphabet_map():
    logging.info("Build alphabet map")
    if exists(METADATA):

        language_map, font_map = parse_metadata(METADATA)

        alphabet_map = {}

        for alphabet_name, font_list in language_map.items():
            logging.info("Check fonts for alphabet %s.", alphabet_name)
            if alphabet_name in SYMBOL_MAP.keys():
                # TODO: after filtering many languages have less fonts. It's expected that all languages with 3-part chars
                #       will fail because it's not currently supported by check_font (e.g., bengali language). However,
                #       even after filtering fonts that don't contain all characters some images fail to render properly.
                #       Use view-dataset on arabic language for an example (bottom row, 4th image from the right). Chars
                #       that fail to render are displayed as # (added this to facilitate spotting them). But be careful,
                #       that # trick doesn't always work and sometimes the image is left blank...
                font_list = [f for f in font_list if
                             f not in FONT_BLACKLIST and check_font(f, SYMBOL_MAP[alphabet_name])]
                alphabet_map[alphabet_name] = Alphabet(alphabet_name, font_list, SYMBOL_MAP[alphabet_name])

        return alphabet_map

    else:
        from warnings import warn
        warn("METADATA file doesn't exists. Most likely, you are not running in the docker."
             "Falling back to default fonts")
        return {'latin': Alphabet('latin', DEFAULT_LATIN_FONT_LIST, SYMBOL_MAP['latin'])}


DEFAULT_LATIN_FONT_LIST = """\
SignPainter
Courrier New
Avenir Next Condensed
Iowan Old Style
Arial
Hiragino Kaku Gothic Pro
Skia
Times
Luminari
Krungthep
Trattatello
Chalkboard
Impact
Snell Roundhand
Brush Script MT
Papyrus
Kefa
Chalkduster
Bodoni 72 Smallcaps
Marker Felt
Apple Chancery
Bradley Hand
DIN Condensed
Zapfino
Sukhumvit Set
Silom
Noteworthy
Phosphate
Copperplate
Futura
Superclarendon
Zapfino
""".splitlines()

ALPHABET_MAP = build_alphabet_map()
