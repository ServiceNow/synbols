#!/usr/bin/env python
"""Script for generating font_property.json when new fonts are added."""

import logging

logging.basicConfig(level=logging.INFO)

from synbols.fonts.google import ALPHABET_MAP
from synbols.fonts.filters import eval_all_font_properties
from synbols.fonts import FONT_PROPERTY_FILE
from os import path
import os
import json

if __name__ == "__main__":

    if path.exists(FONT_PROPERTY_FILE):
        os.rename(FONT_PROPERTY_FILE, FONT_PROPERTY_FILE + '_old')

    property_map = eval_all_font_properties(ALPHABET_MAP, 100)

    with open(FONT_PROPERTY_FILE, 'w') as fd:
        json.dump(property_map, fd)
