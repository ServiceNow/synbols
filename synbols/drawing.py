import os
from glob import glob

import cairo
import numpy as np
from PIL import Image as PILImage


def draw_symbol(ctxt, attributes):
    """Core function drawing the characters as described in `attributes`

    Args:
        ctxt: cairo context to draw the image
        attributes: Object of type Symbol

    Returns:
        extent: rectangle containing the text in the coordinate of the context
        extent_main_char: rectangle containing the central character \
    in the coordinate of the context
    """
    # attributes.background.draw(ctxt)

    attributes.foreground.set_as_source(ctxt)

    weight = cairo.FontWeight.BOLD if attributes.is_bold else cairo.FontWeight.NORMAL
    slant = cairo.FontSlant.OBLIQUE if attributes.is_slant else cairo.FontSlant.NORMAL
    char = attributes.char

    ctxt.set_font_size(1)
    ctxt.select_font_face(attributes.font, cairo.FONT_SLANT_NORMAL, weight)

    extent = ctxt.text_extents(char)
    # normalize font size
    font_size = attributes.scale / max(extent.width, extent.height)
    ctxt.set_font_size(font_size)

    # extent = ctxt.text_extents(char)
    # print(max(extent.width, extent.height))
    # print()
    font_matrix = ctxt.get_font_matrix()

    # set slant to normal and perform it manually.
    # There seems to be some issues with system italic
    if slant != cairo.FONT_SLANT_NORMAL:
        font_matrix = font_matrix.multiply(cairo.Matrix(1, 0.2, 0., 1))

    font_matrix.rotate(attributes.rotation)
    ctxt.set_font_matrix(font_matrix)

    extent = ctxt.text_extents(char)

    translate = (np.array(attributes.translation) + 1.) / 2.
    translate *= np.array((1 - extent.width, 1 - extent.height))
    ctxt.translate(-extent.x_bearing, -extent.y_bearing)
    ctxt.translate(translate[0], translate[1])

    ctxt.show_text(char)

    ctxt.clip()
    ctxt.paint()

    # TODO verify that the extent is the final extent
    # and not the one before translate
    return extent, None


class Pattern(object):
    """Base class for all patterns"""

    def surface(self, width, height):
        surface, ctxt = _make_surface(width, height)
        self.draw(ctxt)
        return surface

    def draw(self, ctxt):
        ctxt.rectangle(0, 0, 1, 1)  # Rectangle(x0, y0, x1, y1)
        self.set_as_source(ctxt)
        ctxt.fill()

    def set_as_source(self, ctxt):
        raise NotImplementedError()

    def attribute_dict(self):
        return {'style': self.__class__.__name__}


class RandomPattern(Pattern):
    """Base class for patterns using a seed."""

    def attribute_dict(self):
        return {'style': self.__class__.__name__,
                'seed': self.seed}


class NoPattern(Pattern):
    def draw(self, ctxt):
        pass

    def set_as_source(self, ctxt):
        ctxt.set_source_rgba(1, 1, 1, 0)


class SolidColor(Pattern):
    """Uses fixed color to render pattern."""

    def __init__(self, color=None):
        self.color = color

    def draw(self, ctxt):
        self.set_as_source(ctxt)
        ctxt.paint()

    def set_as_source(self, ctxt):
        ctxt.set_source_rgb(*self.color)


# There is a plan to make to color sampler a bit more fancy.
def color_sampler(rng=np.random, brightness_range=(0, 1)):
    def sampler():
        b_delta = brightness_range[1] - brightness_range[0]
        return rng.rand(3) * b_delta + brightness_range[0]

    return sampler


class Gradient(RandomPattern):
    """Uses linear or radial graidents to render patterns."""

    def __init__(self,
                 alpha=1,
                 types=('radial', 'linear'),
                 random_color=None,
                 seed=None):
        self.random_color = random_color
        self.seed = seed

        self.types = types
        self.alpha = alpha

    def set_as_source(self, ctxt):
        rng = np.random.RandomState(self.seed)

        pat = _random_pattern(self.alpha,
                              self.random_color,
                              rng=rng,
                              patern_types=self.types)
        ctxt.set_source(pat)


class MultiGradient(RandomPattern):
    """Renders multiple gradient patterns at with transparency."""

    def __init__(self,
                 alpha=0.5,
                 n_gradients=2,
                 types=('radial', 'linear'),
                 random_color=None,
                 seed=None):

        self.random_color = random_color
        self.seed = seed
        self.types = types
        self.alpha = alpha
        self.n_gradients = n_gradients

    def draw(self, ctxt):
        rng = np.random.RandomState(self.seed)
        for i in range(self.n_gradients):
            if i == 0:
                alpha = self.alpha
            else:
                alpha = self.alpha
            pat = _random_pattern(alpha,
                                  self.random_color,
                                  rng=rng,
                                  patern_types=self.types)
            ctxt.rectangle(0, 0, 1, 1)  # Rectangle(x0, y0, x1, y1)
            ctxt.set_source(pat)
            ctxt.fill()

    def set_as_source(self, ctxt):
        raise NotImplementedError()


def _random_pattern(alpha=0.8,
                    random_color=None,
                    patern_types=('linear', 'radial'),
                    rng=np.random):
    """"Select a random pattern with either radial or linear gradient."""
    if random_color is None:
        random_color = color_sampler(rng)
    pattern_type = rng.choice(patern_types)

    # TODO have the locations dependant on the symbol position
    # or in general have a more intelligent design
    if pattern_type == 'linear':
        x1, y1 = rng.rand(2)
        theta = rng.rand() * 2 * np.pi
        r = rng.randn() * 0.2 + 1
        x2 = x1 + r * np.cos(theta)
        y2 = y1 + r * np.sin(theta)
        pat = cairo.LinearGradient(x1, y1, x2, y2)
    elif pattern_type == 'radial':
        x1, y1, x2, y2 = rng.rand(4)
        pat = cairo.RadialGradient(x1, y1, 2, x2, y2, 0.1)
    else:
        raise Exception("unknown pattern type %s" % pattern_type)

    r, g, b = random_color()
    pat.add_color_stop_rgba(1, r, g, b, alpha)
    r, g, b = random_color()
    pat.add_color_stop_rgba(0.5, r, g, b, alpha)
    r, g, b = random_color()
    pat.add_color_stop_rgba(0, r, g, b, alpha)
    return pat


class Camouflage(RandomPattern):
    def __init__(self,
                 stroke_length=0.4,
                 stroke_width=0.05,
                 stroke_angle=np.pi / 4,
                 stroke_noise=0.02,
                 n_stroke=500,
                 seed=None):
        self.seed = seed
        self.stroke_length = stroke_length
        self.stroke_width = stroke_width
        self.n_stroke = n_stroke
        self.stroke_angle = stroke_angle
        self.stroke_noise = stroke_noise

    def draw(self, ctxt):
        stroke_vector = self.stroke_length * \
                        np.array([np.cos(self.stroke_angle), np.sin(self.stroke_angle)])
        rng = np.random.RandomState(self.seed)
        for i in range(self.n_stroke):
            start = (rng.rand(2) * 1.6 - 0.3) * (1 - stroke_vector)
            stop = start + stroke_vector + rng.randn(2) * self.stroke_noise
            ctxt.move_to(*start)
            ctxt.line_to(*stop)
            ctxt.set_line_width(self.stroke_width)

            b, g, r = rng.rand(3)

            ctxt.set_source_rgba(b, g, r, 0.8)
            ctxt.stroke()

    def set_as_source(self, ctxt):
        surface = ctxt.get_group_target()
        width, height = surface.get_width(), surface.get_height()
        source_surface = self.surface(width, height)
        ctxt.set_source_surface(source_surface)

    def to_json(self):
        pass


class ImagePattern(RandomPattern):
    """Uses natural images to render patterns.

    Args:
        root : str, Base path to search for images.
        rotation: float, Maximum random rotation in radian, default 0.
        translation: float, Maximum random translation in proportion, default 1.
        crop: bool, Whether to take a random crop of the image or not, default True.
        min_crop_size: float, Crop's minimal proportion from the image, default 0.2.
        seed : Optional[int], Random seed to use for transformation, default to None
    """

    def __init__(self, root='/images', rotation=0, translation=0.,
                 crop=True, min_crop_size=0.2, seed=None):
        # TODO more extensions
        self._path = glob(os.path.join(root, '**', '*.*'), recursive=True)
        self._path = list(
            filter(lambda p: os.path.splitext(p)[1] in ('.jpg', '.png', '.gif'), self._path))
        self.rotation = rotation
        self.translation = translation
        self.crop = crop
        self.seed = seed
        self.min_crop_size = min_crop_size
        self.rng = np.random.RandomState(self.seed)

    def _rotate_and_translate(self, im, rotation, translation):
        """Randomly rotate and translate the image."""
        w, h = im.size
        rot = np.rad2deg(rotation) * ((self.rng.rand() - 0.5) * 2)
        translation_x = w * translation * ((self.rng.rand() - 0.5) * 2)
        translation_y = h * translation * ((self.rng.rand() - 0.5) * 2)
        return im.rotate(rot, translate=(translation_x, translation_y))

    def _random_crop(self, im, min_crop_size):
        """Randomly crop the image with a minimal crop size of `min_crop_size`% of the image."""
        w, h = im.size
        min_crop_size_x = int(min_crop_size * w)
        min_crop_size_y = int(min_crop_size * h)
        crop_x2 = self.rng.randint(min_crop_size_x + 1, w)
        crop_y2 = self.rng.randint(min_crop_size_y + 1, h)
        x1 = self.rng.randint(0, crop_x2 - min_crop_size_x)
        y1 = self.rng.randint(0, crop_y2 - min_crop_size_y)
        return im.crop((x1, y1, crop_x2, crop_y2))

    def draw(self, ctxt):
        self.set_as_source(ctxt)
        ctxt.paint()

    def set_as_source(self, ctxt):
        surface = ctxt.get_group_target()
        width, height = surface.get_width(), surface.get_height()
        im = PILImage.open(self.rng.choice(self._path, 1).item()).convert('RGB')
        im = self._rotate_and_translate(im, self.rotation, self.translation)
        # Generate a crop with a least 10% of the image in it.
        if self.crop:
            im = self._random_crop(im, min_crop_size=self.min_crop_size)
        im = im.resize((width, height))
        ctxt.set_source_surface(_from_pil(im))


def _surface_to_array(surface):
    """Converts a cairo.ImageSurface object to a numpy array."""
    buf = surface.get_data()
    img = np.ndarray(shape=(surface.get_height(),
                            surface.get_width(),
                            4),
                     dtype=np.uint8,
                     buffer=buf)
    return img[:, :, :3]


def _make_surface(width, height):
    """Creates a cairo.ImageSurface and cairo.Context."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    scale = min(width, height)
    surface.set_device_scale(scale, scale)
    ctxt = cairo.Context(surface)
    return surface, ctxt


def _image_transform(img,
                     inverse_color,
                     pixel_noise_scale,
                     is_gray,
                     max_contrast,
                     rng, symbols=()):
    """Basic array transformation of the image."""
    img = img.astype(np.float32) / 256.

    if is_gray:
        img = np.mean(img, axis=2, keepdims=True)

    if inverse_color:
        img = 1 - img

    if max_contrast:
        mn, mx = np.min(img), np.max(img)

        if mx - mn > 0:
            img = (img - mn) / (mx - mn)
        else:
            if len(symbols) > 0:
                print("Font %s yields empty image" % symbols[0].font)

    img += rng.randn(*img.shape) * pixel_noise_scale
    img = np.clip(img, 0., 1.)

    return (img * 255).astype(np.uint8)


def _from_pil(im, alpha=1.0, format=cairo.FORMAT_ARGB32):
    """ Convert a PIL Image to a Cairo surface.

    Args:
        im: Pillow Image
        alpha: 0..1 alpha to add to non-alpha images
        format: Pixel format for output surface

    Returns: a cairo.ImageSurface object
    """
    assert format in (cairo.FORMAT_RGB24, cairo.FORMAT_ARGB32), f"Unsupported pixel format: {format}"
    if 'A' not in im.getbands():
        im.putalpha(int(alpha * 256.))
    arr = bytearray(im.tobytes('raw', 'BGRa'))
    surface = cairo.ImageSurface.create_for_data(arr, format, im.width, im.height)
    surface.set_device_scale(im.width, im.height)
    return surface


class Image:
    """High level class for genrating an image with symbols, based on attributes.

    Attributes:
        symbols: a list of objects of type Symbol
        resolution: a pair of integer describing the resolution of the image.\
    Defaults to (32, 32).
        background: an object of type Pattern for rendering the background \
    of the image. Defaults to NoPattern.
        inverse_color: Boolean, specifying if the colors should be inverted. \
    Defaults to False.
        pixel_noise_scale: The standard deviation of the pixel noise. \
    Defaults to 0.01.
        max_contrast: Boolean, specifying if the image contrast should \
    be maximized after rendering. If True, the \
    pixel values will be linearly map to range [0, 1] within an image. \
    Defaults to True.
        seed: The random seed of an image. For the same seed, \
    the same image will be rendered. Defaults to None.
    """

    def __init__(self,
                 symbols,
                 resolution=(32, 32),
                 background=NoPattern(),
                 inverse_color=False,
                 pixel_noise_scale=0.01,
                 is_gray=False,
                 max_contrast=True,
                 seed=None):
        self.symbols = symbols
        self.resolution = resolution
        self.inverse_color = inverse_color
        self.pixel_noise_scale = pixel_noise_scale
        self.background = background
        self.is_gray = is_gray
        self.max_contrast = max_contrast
        self.seed = seed

    def make_mask(self):
        mask_list = []
        for symbol in self.symbols:
            mask_list.append(symbol.make_mask(self.resolution))
        return np.concatenate(mask_list, axis=2)

    def make_image(self):
        surface, ctxt = _make_surface(*self.resolution)
        self.background.draw(ctxt)
        for symbol in self.symbols:
            ctxt.save()
            symbol.draw(ctxt)
            ctxt.restore()
        img = _surface_to_array(surface)
        rng = np.random.RandomState(self.seed)
        return _image_transform(img,
                                self.inverse_color,
                                self.pixel_noise_scale,
                                self.is_gray,
                                self.max_contrast,
                                rng, self.symbols)

    def attribute_dict(self):
        symbols = [symbol.attribute_dict() for symbol in self.symbols]
        data = dict(
            resolution=self.resolution,
            pixel_noise_scale=self.pixel_noise_scale,
            seed=self.seed,
            background=self.background.attribute_dict()
        )
        data.update(symbols[0])  # hack to allow flatten access
        data['symbols'] = symbols

        return data

    def add_symbol(self, symbol):
        self.symbols.append(symbol)


class Symbol:
    """Class containing attributes describing each symbol

    Attributes:
        alphabet: Object of type Alphabet
        char: string of 1 or more characters in the image
        font: string describing the font used to draw characters
        foreground: object of type Pattern, \
    used for the foreground of the symbol
        is_slant: bool describing if char is italic or not
        is_bold: bool describing if char is bold or not
        rotation: float, rotation angle of the text
        scale: float, scale of the text. A scale of 1 will have the \
    longest extent of the symbol cover the whole image.
        translation: relative (x, y) translation of the text. \
    A translation in the range [-1, 1] will ensure that the
            symbol fits entirely in the image. Note if the scale i
    """

    def __init__(self,
                 alphabet,
                 char,
                 font,
                 foreground,
                 is_slant,
                 is_bold,
                 rotation,
                 scale,
                 translation):
        self.alphabet = alphabet
        self.char = char
        self.font = font
        self.is_bold = is_bold
        self.is_slant = is_slant
        self.foreground = foreground
        self.rotation = rotation
        self.scale = scale
        self.translation = translation

    def draw(self, ctxt):
        draw_symbol(ctxt, self)

    def make_mask(self, resolution):
        """Creates a grey scale image
        corresponding to the mask of the symbol.
        """
        fg = self.foreground
        self.foreground = SolidColor((1, 1, 1))
        surface, ctxt = _make_surface(*resolution)
        draw_symbol(ctxt, self)
        self.foreground = fg
        img = _surface_to_array(surface)
        return np.mean(img, axis=2, keepdims=True).astype(np.uint8)

    def attribute_dict(self):
        """Returns a dict of all attributes of the symbol."""
        return dict(
            alphabet=self.alphabet.name,
            char=self.char,
            font=self.font,
            is_bold=str(self.is_bold),
            is_slant=str(self.is_slant),
            scale=self.scale,
            translation=self.translation,
            rotation=self.rotation,
            foreground=self.foreground.attribute_dict(),
        )
