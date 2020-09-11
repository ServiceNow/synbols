from tqdm import tqdm
import numpy as np
import types
from .drawing import Gradient, Image, Symbol
from .data_io import write_h5
from .fonts import LANGUAGE_MAP
from .utils import make_img_grid


def _select(default, value, rng):
    if value is None:
        return default(rng)
    elif callable(value):
        return value(rng)
    else:
        return value


def _rand_seed(rng):
    return rng.randint(np.iinfo(np.uint32).max)


# ya basic!
def basic_attribute_sampler(alphabet=None, char=None, font=None, background=None,
                            foreground=None, is_slant=None, is_bold=None, rotation=None, scale=None, translation=None,
                            inverse_color=None, max_contrast=None, pixel_noise_scale=None, resolution=(32, 32),
                            is_gray=False, n_symbols=1):
    """Returns a function that generates a new Image object on every call.

    This function is the high level interface for defining a new distribution over images. On every call, it will return
    an drawing.Image object, containing every attributes to render the final image into a numpy array. All arguments
    to this function have a proper default value. When no argument are passed, this is referred to as the
    "default" synbols dataset.

    All arguments can be either a constant, a callable, or None. If None is passed, the default distribution is used.
    A callable can be used to define a distribution over the specific argument. This function must take 1 argument
    specifying the random number generator.

    Args:
        alphabet: Object of type utils.Alphabet or a distribution over it.
            An alphabet can be created easily using
            Language.get_alphabet(). This argument is only used to specify the default distributions over char and
            fonts. If these arguments are specified, alphabet is ignored.
        char: string or distribution over strings.
            Defaults to Uniform(alphabet.symbols)
        font: string or distribution over strings.
            Defaults to Uniform(alphabet.fonts)
        background: object of type drawing.Pattern or distribution over it.
            Defines how the background will be rendered. Defaults to drawing.Gradient
        foreground: object of type drawing.Pattern or distribution over it.
            Defines how the foreground will be rendered. Defaults to drawing.Gradient
        is_slant: bool or distribution over bool.
            Defines if character is drawn italic or normal. For wider support, this is done using the a 2D
            transformation instead of relying on the font's italic. Defaults to Uniform{True, False}.
        is_bold: bool or distribution over bool.
            Whether the character is rendered in bold or not. Note: Some fonts do not support boldd. In which case, it
            will have no effect. To obtain a collection of font that support bold, use
            Language.get_alphabet(... support_bold=True)
        rotation: float or distribution over it.
            Rotation of the symbol in radian in the range [-pi .. pi]. Defaults to Normal(0, 0.3).
        scale: float or distribution over it.
            Scale of the symbol. A scale of 1 will have either the width or height cover the whole image. Defaults to
            0.6* exp(Normal(0, 0.2))
        translation: a pair of float or a distribution over it.
            Numbers between [-1 .. 1] will make sure the symbol
            stays withing the image i.e. the actual translation depends on the remaining space after the symbol is
            scaled. Defaults to Uniform(-1, 1).
        inverse_color: bool or a distribution over it.
            If True, returns 1 - pixel_value to inverse the value of all
            pixels. Defaults to Uniform([True, False])
        max_contrast: bool or distribution over it.
            If True, pixel values will be rescaled to span 0..1 inside each
            image. Defaults to True.
        pixel_noise_scale: float or a distribution over it.
            The standard deviation of the pixel noise. Defaults to 0.01.
        resolution: A pair of integer.
            Defines the resolution of the image. Defaults to (32, 32).
        is_gray: bool.
            If True, the color channels are averaged into a single channel. Defaults to False.
        n_symbols: integer or a distribution over it.
            Number of symbols to rendered in the image. All arguments that are distributions will be sampled multiple
            times to provide different symbols. Defaults to 1. Note: if the number of symbols is variable, you will have
            to provide a proper mask_aggregator when calling dataset_generator e.g. flatten_mask.
    Returns:
        A callable taking an optional seed as an argument and returning an object of type drawing.Image.
    """

    def sampler(seed=None):
        _rng = np.random.RandomState(seed)
        symbols = []
        _n_symbols = _select(1, n_symbols, _rng)
        for i in range(_n_symbols):
            _alphabet = _select(lambda rng: LANGUAGE_MAP['english'].get_alphabet(support_bold=True), alphabet, _rng)

            _char = _select(lambda rng: rng.choice(_alphabet.symbols), char, _rng)
            _font = _select(lambda rng: rng.choice(sorted(_alphabet.fonts)), font, _rng)
            _is_bold = _select(lambda rng: rng.choice([True, False]), is_bold, _rng)
            _is_slant = _select(lambda rng: rng.choice([True, False]), is_slant, _rng)
            _rotation = _select(lambda rng: rng.randn() * 0.3, rotation, _rng)
            _scale = _select(lambda rng: 0.6 * np.exp(rng.randn() * 0.2), scale, _rng)
            _translation = _select(lambda rng: tuple(rng.rand(2) * 1.8 - 0.9), translation, _rng)
            _foreground = _select(lambda rng: Gradient(seed=_rand_seed(_rng)), foreground, _rng)

            symbols.append(Symbol(alphabet=_alphabet, char=_char, font=_font, foreground=_foreground,
                                  is_slant=_is_slant, is_bold=_is_bold, rotation=_rotation, scale=_scale,
                                  translation=_translation))

        _background = _select(
            lambda rng: Gradient(seed=_rand_seed(_rng)), background, _rng
        )
        _inverse_color = _select(
            lambda rng: rng.choice([True, False]), inverse_color, _rng
        )
        _pixel_noise_scale = _select(
            lambda rng: 0.01, pixel_noise_scale, _rng
        )
        _max_contrast = _select(
            lambda rng: True, max_contrast, _rng
        )

        return Image(symbols,
                     background=_background,
                     inverse_color=_inverse_color,
                     resolution=resolution,
                     pixel_noise_scale=_pixel_noise_scale,
                     is_gray=is_gray,
                     max_contrast=_max_contrast,
                     seed=_rand_seed(_rng))

    return sampler


def flatten_mask(masks):
    overlap = np.mean(masks.sum(axis=-1) >= 256)

    flat_mask = np.zeros(masks.shape[:-1])

    for i in range(masks.shape[-1]):
        flat_mask[(masks[:, :, i] > 2)] = i + 1
    return flat_mask, {'overlap_score': overlap}


def flatten_mask_except_first(masks):
    return np.stack((masks[:, :, 0], flatten_mask(masks[:, :, 1:])[0]), axis=2)


def add_occlusion(attr_sampler,
                  n_occlusion=None,
                  occlusion_char=None,
                  rotation=None,
                  scale=None,
                  translation=None,
                  foreground=None):

    """Augment an attribute sampler to add occlusions over the other symbols.

    Args:
        attr_sampler: a callable returning an object of type drawing.Image.
        n_occlusion: integer or a distribution over it.
            Specifies the number of occlusions to draw. Defaults to Uniform([1 .. 5])
        occlusion_char: string or distribution over it.
            Specifies the unicode symbols used to make occlusions. Defaults to Uniform(['■', '▲', '●']).
        rotation: float or distribution over it.
            Rotation of the symbol in radian in the range [-pi .. pi]. Defaults to Uniform([-pi .. pi]).
        scale: float or distribution over it.
            Scale of the symbol. A scale of 1 will have either the width or height cover the whole image. Defaults to
            0.3* exp(Normal(0, 0.1))
        translation: a pair of float or a distribution over it.
            Numbers between [-1 .. 1] will make sure the symbol
            stays withing the image i.e. the actual translation depends on the remaining space after the symbol is
            scaled. Defaults to Uniform(-1.5, 1.5).
        foreground: object of type drawing.Pattern or distribution over it.
            Defines how the foreground will be rendered. Defaults to drawing.Gradient
    Returns:
        A callable taking an optional seed as an argument and returning an object of type drawing.Image.
    """
    occlusion_chars = ['■', '▲', '●']

    def sampler(seed=None):
        image = attr_sampler(seed)
        _rng = np.random.RandomState(seed)
        _n_occlusion = _select(
            lambda rng: rng.randint(1, 5), n_occlusion, _rng
        )

        for i in range(_n_occlusion):
            _scale = _select(lambda rng: 0.3 * np.exp(rng.randn() * 0.1), scale, _rng)
            _translation = _select(lambda rng: tuple(rng.rand(2) * 3 - 1.5), translation, _rng)

            _occlusion_char = _select(lambda rng: rng.choice(occlusion_chars), occlusion_char, _rng)
            _rotation = _select(lambda rng: rng.rand() * np.pi * 2, rotation, _rng)
            _foreground = _select(lambda rng: Gradient(seed=_rand_seed(_rng)), foreground, _rng)

            occlusion = Symbol(LANGUAGE_MAP['english'].get_alphabet(), _occlusion_char, font='Arial',
                               foreground=_foreground, rotation=_rotation, scale=_scale, translation=_translation,
                               is_slant=False, is_bold=False)
            image.add_symbol(occlusion)

        return image

    return sampler


def dataset_generator(attr_sampler,
                      n_samples,
                      mask_aggregator=None,
                      seed=None):
    """High level function generating the dataset from an attribute sampler."""

    if isinstance(attr_sampler, types.GeneratorType):
        attr_generator = attr_sampler

        def sampler(_seed=None):  # ignores the seed
            return next(attr_generator)

        attr_sampler = sampler

    rng = np.random.RandomState(seed)

    for i in tqdm(range(n_samples)):
        attributes = attr_sampler(_rand_seed(rng))
        mask = attributes.make_mask()
        x = attributes.make_image()
        y = attributes.attribute_dict()

        if mask_aggregator is not None:
            mask = mask_aggregator(mask)
            if isinstance(mask, tuple):
                mask, mask_attributes = mask
                y.update(mask_attributes)

        yield x, mask, y


def generate_and_write_dataset(file_path, attr_sampler, n_samples, preview_shape=(10, 10), seed=None):
    """Call the attribute sampler n_samples time to generate a dataset and saves it on disk.

    Args:
        file_path: the destination of the dataset an extension .h5py will be automatically added.
        attr_sampler: a callable returning objects of type drawing.Image.
        n_samples: integer specifying the number of samples required.
        preview_shape: pair of integers or None. Specifies the size of the image grid to render a preview. The png
            will be saved alongside the dataset.
        seed: integer or None. Specifies the seed the random number generator.
    """
    ds_generator = dataset_generator(attr_sampler, n_samples, seed=seed)

    if preview_shape is not None:
        n_row, n_col = preview_shape
        ds_generator = make_preview(ds_generator,
                                    file_path + "_preview.png",
                                    n_row=n_row,
                                    n_col=n_col)

    write_h5(file_path + ".h5py", ds_generator, n_samples)


def make_preview(generator, file_name, n_row=10, n_col=10):
    """Augment a generator to save a preview when the first n_row * n_col images are generated."""
    x_list = []
    y_list = []
    for x, mask, y in generator:

        if x_list is not None:

            x_list.append(x)
            y_list.append(y)

            if len(x_list) == n_row * n_col:
                from PIL import Image
                from scipy.ndimage import zoom
                img_grid, _, _ = make_img_grid(
                    np.stack(x_list),
                    y_list,
                    h_axis=None,
                    v_axis=None,
                    n_row=n_row,
                    n_col=n_col)

                # zoom by a factor of 2 to be able to see
                # the pixelization through viewers that use bicubic zooming
                zoom_factor = (2, 2, 1) if img_grid.ndim == 3 else (2, 2)
                img_grid = zoom(img_grid, zoom_factor, order=0)

                Image.fromarray(img_grid).save(file_name)

                x_list = None
                tqdm.write("Preview generated.")

        yield x, mask, y


def generate_char_grid(language, n_char, n_font, seed=None, **kwargs):
    """Generate a dense grid of n_char x n_font.
    Mainly for visualization purpose.
    """

    def _attr_generator():
        alphabet = LANGUAGE_MAP[language].get_alphabet()
        rng = np.random.RandomState(seed)
        chars = rng.choice(alphabet.symbols, n_char, replace=False)
        fonts = rng.choice(alphabet.fonts, n_font, replace=False)
        for char in chars:
            for font in fonts:
                yield basic_attribute_sampler(alphabet,
                                              char=char,
                                              font=font,
                                              **kwargs)(_rand_seed(rng))

    return dataset_generator(_attr_generator(), n_char * n_font, flatten_mask)


def text_generator(char_list, seed=None, **kwargs):
    """Generate a string of synbols. Mainly for advertisement purpose"""
    rng = np.random.RandomState(seed)

    def _attr_generator():
        for char in char_list:
            yield basic_attribute_sampler(char=char, **kwargs)(_rand_seed(rng))

    return dataset_generator(_attr_generator(), len(char_list))
