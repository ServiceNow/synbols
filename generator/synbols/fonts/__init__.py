import os
import json
from os import path
from warnings import warn

from ..utils import Alphabet, SYMBOL_MAP

FONT_PROPERTY_FILE = path.join(path.dirname(__file__), 'font_property.json')
FONT_CLUSTERS_FILE = path.join(path.dirname(__file__), 'hierarchical_clustering_font.json')

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


try:
    from .google import ALPHABET_MAP

    if os.path.exists(FONT_PROPERTY_FILE):
        with open(FONT_PROPERTY_FILE, "r") as fd:
            font_properties = json.load(fd)
    else:
        raise Exception("""File %s doesn't exist, please run the script "font_checks". """ % FONT_PROPERTY_FILE)

    if os.path.exists(FONT_CLUSTERS_FILE):
        with open(FONT_CLUSTERS_FILE, "r") as fd:
            font_clusters = {'latin': json.load(fd)}
    else:
        warn("""File %s doesn't exist, font clustering will be ignored """ % FONT_CLUSTERS_FILE)
        font_clusters = {}

    for alphabet in ALPHABET_MAP.values():
        alphabet.filter_fonts(font_properties.get(alphabet.name), font_clusters.get(alphabet.name))

except FileNotFoundError:
    # Fallback to a default alphabet map
    # XXX: This should happen when running the code locally (not in the Docker image)

    warn("Google fonts not found. Most likely, you are not running in the Docker image."
         "Falling back to default fonts")
    ALPHABET_MAP = {'latin': Alphabet('latin', DEFAULT_LATIN_FONT_LIST, SYMBOL_MAP['latin'])}
