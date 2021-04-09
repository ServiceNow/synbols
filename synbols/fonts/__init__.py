from warnings import warn

from ..utils import load_all_languages


# Load all support languages (characters and fonts)
LANGUAGE_MAP = load_all_languages()
if len(LANGUAGE_MAP) == 0:
    warn(
        "No locale files were found. \
    Did you execute the code with the 'synbols' executable?"
    )
