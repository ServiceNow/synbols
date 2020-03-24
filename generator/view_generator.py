from synbols.data_io import pack_dataset
from synbols.drawing import Camouflage
from synbols.generate import generate_char_grid
from view_dataset import plot_dataset
import synbols
import numpy as np

if __name__ == "__main__":
    alphabet = 'latin'  # TODO missing upper cases
    # alphabet = 'telugu'  # TODO the circle in some chars is not rendered. (bottom ones)
    # alphabet = 'thai'
    # alphabet = 'vietnamese'
    # alphabet = 'arabic'  # TODO: missing chars in fonts
    # alphabet = 'hebrew'
    # alphabet = 'khmer' # TODO: see comment in googlefonts
    # alphabet = 'tamil'
    # alphabet = 'gujarati' # TODO: one font to remove and one blank in another font
    # alphabet = 'bengali'
    # alphabet = 'malayalam'
    # alphabet = 'korean' # TODO: huge amount of missing chars
    # alphabet = 'chinese-simplified'
    # alphabet = 'greek'
    # alphabet = 'cyrillic'


    fg = Camouflage(stroke_angle=0.5)
    bg = Camouflage(stroke_angle=1.)

    ## Uncomment to remove background
    # fg, bg = None, None

    ## Uncomment for a smoother background
    fg, bg = 'gradient', 'gradient'

    kwargs = dict(foreground=fg, background=bg, is_bold=True, scale=(1.3, 1.3),
                  resolution=(32, 32), rng=42)

    x, y = pack_dataset(generate_char_grid('latin', n_font=2, n_char=10, **kwargs))
    plot_dataset(x, y, name=alphabet, h_axis='char', v_axis='font', rng=np.random.RandomState(42))
