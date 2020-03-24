import os
import pickle

from .filters import filter_fonts

CACHE_FILE = "alphabet_fonts.cache"


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