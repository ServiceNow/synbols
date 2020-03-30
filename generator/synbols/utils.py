import numpy as np


class Alphabet:
    """Combines fonts and symbols for a given language."""

    def __init__(self, name, fonts, symbols):
        self.name = name
        self.symbols = symbols
        self.fonts = fonts
#
#
# def _check_random_state(rng):
#     if isinstance(rng, np.random.RandomState):
#         return rng
#     else:
#         return np.random.RandomState(rng)