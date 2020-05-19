import numpy as np
import logging
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO)

from synbols.data_io import pack_dataset
from synbols.drawing import Camouflage, color_sampler, Gradient, MultiGradient, NoPattern, SolidColor
from synbols.generate import generate_char_grid, dataset_generator, basic_image_sampler, add_occlusion, \
    flatten_mask_except_first
from synbols.fonts import ALPHABET_MAP

from view_dataset import plot_dataset
import matplotlib.pyplot as plt

font_list = """\
jotione
lovedbytheking
flavors
mrbedfort
butterflykids
newrocker
smokum
jimnightshade
""".splitlines()


def make_image(attr_sampler, file_name):
    x, _, y = pack_dataset(dataset_generator(attr_sampler, 1000))

    plot_dataset(x, y, h_axis='font', v_axis='char')

    plt.savefig(file_name)


def savefig(file_name):
    plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0)


def show_fonts(seed):
    rng = np.random.RandomState(seed)

    def attr_sampler():
        for char in 'abCD':
            for font in font_list:
                yield basic_image_sampler(
                    alphabet=ALPHABET_MAP['latin'], char=char, font=font, is_bold=False, is_slant=False,
                    resolution=(128, 128), pixel_noise_scale=0, rng=rng)()

    x, _, y = pack_dataset(dataset_generator(attr_sampler().__next__, 1000))
    plot_dataset(x, y, h_axis='font', v_axis='char')

    # savefig('fonts.png')


def show_languages(seed):
    language_list = ['korean',
                     'chinese-simplified',
                     'telugu',
                     'thai',
                     'gujarati',
                     'arabic',
                     'tamil',
                     'hebrew']

    rng = np.random.RandomState(seed)

    def attr_sampler():
        for lang in language_list:
            alphabet = ALPHABET_MAP[lang]
            for i in range(4):
                yield basic_image_sampler(
                    alphabet=alphabet, char=rng.choice(alphabet.symbols), font=rng.choice(alphabet.fonts),
                    is_bold=False, is_slant=False, resolution=(128, 128), pixel_noise_scale=0, rng=rng)()

    x, _, y = pack_dataset(dataset_generator(attr_sampler().__next__, 1000))
    h_values, v_values = plot_dataset(x, y, h_axis='alphabet', v_axis=None, n_col=len(language_list), n_row=4)

    map = {'chinese-simplified': 'chinese'}
    h_values = [map.get(val, val) for val in h_values]

    ax = plt.gca()
    ax.set_xticks((np.arange(len(h_values)) + 0.5) * x.shape[1])
    ax.set_xticklabels(h_values, rotation=0)
    ax.get_xaxis().set_visible(True)
    plt.xlabel('')

    # savefig('language.png')


def show_background(seed):
    rng = np.random.RandomState(seed)
    kwargs = dict(rng=rng, resolution=(128, 128), alphabet=ALPHABET_MAP['latin'], char='a', inverse_color=False,
                  pixel_noise_scale=0)
    graident = Gradient(types=('radial',), rng=rng)
    attr_list = [
        basic_image_sampler(background=SolidColor((0.2, 0.2, 0)), foreground=SolidColor((0.8, 0, 0.8)), **kwargs),
        basic_image_sampler(background=graident, foreground=graident, **kwargs),
        basic_image_sampler(background=Camouflage(stroke_angle=np.pi / 4, rng=rng),
                            foreground=Camouflage(stroke_angle=np.pi * 3 / 4, rng=rng), **kwargs),
        add_occlusion(basic_image_sampler(**kwargs), n_occlusion=3,
                      scale=lambda _: 0.3 * np.exp(rng.randn() * 0.1),
                      translation=lambda _: tuple(rng.rand(2) * 2 - 1),
                      rng=rng)
    ]

    def attr_sampler():
        for attr in attr_list:
            yield attr()

    x, _, y = pack_dataset(dataset_generator(attr_sampler().__next__, 1000, flatten_mask_except_first))
    plot_dataset(x, y, h_axis='scale', v_axis=None, n_col=4, n_row=1)

    ax = plt.gca()
    ax.set_xticks((np.arange(4) + 0.5) * x.shape[1])
    ax.set_xticklabels(['Solid', 'Gradient', 'Camouflage', 'Occlusions'], rotation=0)
    ax.get_xaxis().set_visible(True)
    plt.xlabel('')

    # savefig('background.png')


def pack_dataset_resample(generator, resolution=64):
    """Turn a the output of a generator of (x,y) pairs into a numpy array containing the full dataset"""
    x, mask, y = zip(*generator)
    x = [zoom(img, (resolution / img.shape[0],) * 2 + (1,), order=0) for img in x]
    return np.stack(x), y


def show_resolution(seed):
    kwargs = dict(rng=np.random.RandomState(seed), alphabet=ALPHABET_MAP['latin'], is_bold=False, is_slant=False,
                  inverse_color=False, pixel_noise_scale=0)
    attr_list = [
        basic_image_sampler(resolution=(8, 8), char='b', font='arial', scale=0.9, rotation=0,
                            background=SolidColor((0, 0, 0)),
                            foreground=SolidColor((0.5, 0.5, 0)), **kwargs),
        basic_image_sampler(resolution=(16, 16), char='x', font='time', scale=0.7, **kwargs),
        basic_image_sampler(resolution=(32, 32), char='g', font='flavors', scale=0.6, rotation=1, **kwargs),
        basic_image_sampler(resolution=(64, 64), scale=0.3, n_symbols=5, **kwargs),
    ]

    def attr_sampler():
        for attr in attr_list:
            yield attr()

    x, y = pack_dataset_resample(dataset_generator(attr_sampler().__next__, 1000))
    plot_dataset(x, y, h_axis='rotation', v_axis=None, n_col=4, n_row=1)

    ax = plt.gca()
    ax.set_xticks((np.arange(len(attr_list)) + 0.5) * x.shape[1])
    ax.set_xticklabels(['8 x 8', '16 x 16', '32 x 32', '64 x 64'], rotation=0)
    ax.get_xaxis().set_visible(True)
    plt.xlabel('')

    # savefig('resolution.png')


def alphabet_sizes():
    for name, alphabet in ALPHABET_MAP.items():
        print(name, len(alphabet.symbols))


if __name__ == "__main__":
    # plt.figure('languages', figsize=(5, 3))
    # show_languages()
    #
    # plt.figure('fonts', figsize=(5, 3))
    # show_fonts()
    #
    # plt.figure('resolution', figsize=(5, 3))
    # show_resolution()
    #
    # plt.figure('background', figsize=(5, 3))
    # show_background()

    # alphabet_sizes()

    for i in range(4):
        plt.figure('group %d' % i, figsize=(10, 6))

        plt.subplot(2, 2, 1)
        show_fonts(0)
        plt.title('a) fonts')

        # plt.subplot(2, 2, 2)
        # show_languages(5)
        # plt.title('b) languages')

        plt.subplot(2, 2, 3)
        show_resolution(0)
        plt.title('c) resolution')

        plt.subplot(2, 2, 4)
        show_background(i)
        plt.title('d) background and foreground')

        savefig('group %d.png' % i)
    plt.show()
