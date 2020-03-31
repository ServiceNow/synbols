import os
import pickle

from .filters import filter_fonts
from ..utils import Alphabet, SYMBOL_MAP


CACHE_FILE = "alphabet_fonts.cache"
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
    # Build the list of alphabets and supported fonts (use cache if available)
    if os.path.exists(CACHE_FILE):
        ALPHABET_MAP = pickle.load(open(CACHE_FILE, "rb"))
    else:
        print("Building alphabet/font cache. This may take a few minutes...")
        from .google import ALPHABET_MAP as ALPHABET_MAP_
        ALPHABET_MAP = dict(ALPHABET_MAP_)
        for alphabet in ALPHABET_MAP.values():
            alphabet.fonts = filter_fonts(alphabet)[0]
        pickle.dump(ALPHABET_MAP, open(CACHE_FILE, "wb"))
        del ALPHABET_MAP_
        
except FileNotFoundError:
    # Fallback to a default alphabet map
    # XXX: This should happen when running the code locally (not in the Docker image)
    from warnings import warn
    warn("Google fonts not found. Most likely, you are not running in the Docker image."
         "Falling back to default fonts")
    ALPHABET_MAP = {'latin': Alphabet('latin', DEFAULT_LATIN_FONT_LIST, SYMBOL_MAP['latin'])}