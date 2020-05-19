import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

from synbols.data_io import pack_dataset
from synbols.drawing import Camouflage, color_sampler, Gradient, MultiGradient, NoPattern, SolidColor
from synbols.generate import generate_char_grid, dataset_generator, basic_image_sampler, add_occlusion, \
    flatten_mask_except_first, generate_counting_dataset
from synbols.fonts import ALPHABET_MAP
from  synbols import generate

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

    attr_sampler = basic_image_sampler(alphabet=ALPHABET_MAP['latin'])
    # attr_sampler = add_occlusion(attr_sampler)
    # x, mask, y = pack_dataset(dataset_generator(attr_sampler, 1000, flatten_mask_except_first))
    x, mask, y = pack_dataset(generate_counting_dataset(1000))
    # x, mask, y = pack_dataset(generate_char_grid('latin', n_font=15, n_char=20))


    # mask = [attr['overlap_score'] == 0 for attr in y]
    # y = np.array(y)[mask]
    # x = x[mask]


    plt.figure('dataset')
    plot_dataset(x, y, h_axis=None, v_axis=None, rng=np.random.RandomState(42))

    # print(mask.shape, x.shape)
    # mask = mask.astype(np.float) / 256
    # plt.figure('mask')
    # plot_dataset(mask, y, h_axis='char', v_axis=None, rng=np.random.RandomState(42))

    plt.savefig('tst.png')
    plt.show()
