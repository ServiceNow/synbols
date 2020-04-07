import os
import logging
import json

from .filters import filter_fonts
from ..utils import Alphabet, SYMBOL_MAP

CACHE_FILE = "font_whitelist.json"
DEFAULT_LATIN_FONT_LIST = \
    """
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

    # Build the list of alphabets and supported fonts (use cache if available)
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as fd:
            whitelist = json.load(fd)
    else:
        logging.info("Building alphabet/font cache. This may take a few minutes...")

        del ALPHABET_MAP['korean']
        del ALPHABET_MAP['bengali']

        whitelist = {}
        for i, alphabet in enumerate(ALPHABET_MAP.values()):
            logging.info("alphabet %s (%d / %d)" % (alphabet.name, i + 1, len(ALPHABET_MAP)))
            whitelist[alphabet.name] = filter_fonts(alphabet)[0]

        with open(CACHE_FILE, "w") as fd:
            json.dump(whitelist, fd)

    for name, font_list in whitelist.items():
        ALPHABET_MAP[name].fonts = font_list

except FileNotFoundError:
    # Fallback to a default alphabet map
    # XXX: This should happen when running the code locally (not in the Docker image)
    from warnings import warn

    warn("Google fonts not found. Most likely, you are not running in the Docker image."
         "Falling back to default fonts")
    ALPHABET_MAP = {'latin': Alphabet('latin', DEFAULT_LATIN_FONT_LIST, SYMBOL_MAP['latin'])}
