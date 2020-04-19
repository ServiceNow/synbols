#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
from os.path import join
from os import path
import json
import numpy as np
import logging

from ..utils import Alphabet, SYMBOL_MAP

GOOGLE_FONTS_PATH = "/usr/share/fonts/truetype/google-fonts/"
GOOGLE_FONTS_METADATA_PATH = join(GOOGLE_FONTS_PATH, "google_fonts_metadata")
FONT_BLACKLIST = ["rubik", "podkova", "baloochettan2", "seymourone", "kumarone", "stalinone", "oranienbaum",
                  "stalinistone", "vampiroone"]
FONT_CLUSTERS_PATH = path.join(path.dirname(__file__), 'hierarchical_clustering_font.json')


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


def _blacklist_from_cluster():
    """Blacklist every redundant font in clusters (keep the first one of each cluster)"""
    if path.exists(FONT_CLUSTERS_PATH):
        logging.info('loading blacklist')
        with open(FONT_CLUSTERS_PATH) as fd:
            clusters = json.load(fd)
            for cluster in clusters:
                font_names, values = zip(*cluster)
                main_font_idx = np.argmin(values)
                for i, font_name in enumerate(font_names):
                    if i == main_font_idx:
                        pass
                        logging.info("keeping %s", font_name)
                    else:
                        FONT_BLACKLIST.append(font_name)
                        logging.info("blacklisting %s", font_name)


def build_alphabet_map():
    _blacklist_from_cluster()
    language_map, font_map = parse_metadata(GOOGLE_FONTS_METADATA_PATH)
    alphabet_map = {}
    for alphabet_name, font_list in list(language_map.items()):
        if alphabet_name in SYMBOL_MAP.keys():
            fonts = [f for f in font_list if f not in FONT_BLACKLIST]
            alphabet_map[alphabet_name] = Alphabet(alphabet_name, fonts, SYMBOL_MAP[alphabet_name])
    return alphabet_map


ALPHABET_MAP = build_alphabet_map()
