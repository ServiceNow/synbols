#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from icu import LocaleData
from os.path import join, exists
import logging

from ..utils import Alphabet


FONT_PATH = "/usr/share/fonts/truetype/google-fonts/"
METADATA = join(FONT_PATH, "google_fonts_metadata")


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


def build_alphabet_map():
    logging.info("Build alphabet map")
    language_map, font_map = parse_metadata(METADATA)
    alphabet_map = {}
    for alphabet_name, font_list in list(language_map.items())[:5]:
        logging.info("Check fonts for alphabet %s.", alphabet_name)
        if alphabet_name in SYMBOL_MAP.keys():
            alphabet_map[alphabet_name] = Alphabet(alphabet_name, font_list, SYMBOL_MAP[alphabet_name])
    return alphabet_map


ALPHABET_MAP = build_alphabet_map()