import numpy as np

from synbols.data_io import pack_dataset
from synbols.drawing import Camouflage, color_sampler, Gradient, MultiGradient, NoPattern, SolidColor
from synbols.generate import generate_char_grid
from synbols.fonts import ALPHABET_MAP

from view_dataset import plot_dataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # bg = Camouflage(stroke_angle=1.)
    # bg = NoPattern()
    bg = MultiGradient(alpha=0.5, n_gradients=2, types=('linear', 'radial'))
    # bg = Gradient(types=('linear',), random_color=color_sampler(brightness_range=(0.1, 0.9)))

    # fg = Camouflage(stroke_angle=0.5)
    # fg = SolidColor((1, 1, 1))
    fg = Gradient(types=('radial',), random_color=color_sampler(brightness_range=(0.1, 0.9)))
    # fg = NoPattern()

    kwargs = dict(foreground=fg, background=bg, resolution=(64, 64), n_symbols=3,
                  scale=lambda rng: 0.5 * np.exp(rng.randn() * 0.2),
                  is_bold=False)

    # kwargs = dict()

    x, mask, y = pack_dataset(generate_char_grid('latin', n_font=15, n_char=20, **kwargs))
    print(mask.shape, x.shape)
    mask = mask.astype(np.float)/256

    plot_dataset(x, y, name='latin', h_axis='char', v_axis='font', rng=np.random.RandomState(42))
    plot_dataset(mask, y, name='mask', h_axis='char', v_axis='font', rng=np.random.RandomState(42))

    plt.show()
