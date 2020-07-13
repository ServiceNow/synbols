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

    attr_sampler = basic_image_sampler(alphabet=ALPHABET_MAP['latin'], scale=0.1, n_symbols=50, resolution=(128, 128))
    # attr_sampler = add_occlusion(attr_sampler)
    # x, mask, y = pack_dataset(dataset_generator(attr_sampler, 200, generate.flatten_mask))
    # x, mask, y = pack_dataset(generate.generate_counting_dataset_crowded(200))
    x, mask, y = pack_dataset(generate.generate_korean_1k_dataset(200))

    # x, mask, y = pack_dataset(generate.generate_camouflage_dataset(200))

    # x, mask, y = pack_dataset(generate_char_grid('latin', n_font=15, n_char=20))


    # mask = [attr['overlap_score'] > 0.005 for attr in y]
    # y = np.array(y)[mask]
    # x = x[mask]


    plt.figure('dataset')
    plot_dataset(x, y, h_axis=None, v_axis='font', rng=np.random.RandomState(42), n_row=10, n_col=20)

    # print(mask.shape, x.shape)
    # mask = mask.astype(np.float) / 256
    # plt.figure('mask')
    # plot_dataset(mask, y, h_axis='char', v_axis=None, rng=np.random.RandomState(42))

    plt.savefig('tst.png')
    plt.show()
