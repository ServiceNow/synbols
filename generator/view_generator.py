import numpy as np

from synbols.data_io import pack_dataset
from synbols.drawing import Camouflage, color_sampler, Gradient, MultiGradient, NoPattern, SolidColor
from synbols.generate import generate_char_grid
from view_dataset import plot_dataset
import matplotlib.pyplot as plt

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

    # bg = Camouflage(stroke_angle=1.)
    # bg = NoPattern()
    bg = MultiGradient(alpha=0.5, n_gradients=2, types=('linear', 'radial'))
    # bg = Gradient(types=('linear',), random_color=color_sampler(brightness_range=(0.1, 0.9)))

    # fg = Camouflage(stroke_angle=0.5)
    # fg = SolidColor((1, 1, 1))
    fg = Gradient(types=('radial',), random_color=color_sampler(brightness_range=(0.1, 0.9)))
    # fg = NoPattern()

    kwargs = dict(foreground=fg, background=bg, resolution=(32, 32), n_symbols=1)

    # kwargs = dict()

    x, y = pack_dataset(generate_char_grid('latin', n_font=15, n_char=20, **kwargs))
    plot_dataset(x, y, name=alphabet, h_axis='char', v_axis='font', rng=np.random.RandomState(42))
    plt.show()
