import cairo
import numpy as np
import warnings

from sys import stdout


# from .utils import _check_random_state

#
# SLANT_MAP = {
#     cairo.FONT_SLANT_ITALIC: 'italic',
#     cairo.FONT_SLANT_NORMAL: 'normal',
#     cairo.FONT_SLANT_OBLIQUE: 'oblique',
# }


def draw_symbol(ctxt, attributes):
    """Core function drawing the characters as described in `attributes`

    Args:
        ctxt: cairo context to draw the image
        attributes: Object of type Symbol containing information about the image

    Returns:
        extent: rectangle containing the text in the coordinate of the context
        extent_main_char: rectangle containing the central character in the coordinate of the context
    """
    # attributes.background.draw(ctxt)

    attributes.foreground.set_as_source(ctxt)

    weight = cairo.FontWeight.BOLD if attributes.is_bold else cairo.FontWeight.NORMAL
    slant = cairo.FontSlant.OBLIQUE if attributes.is_slant else cairo.FontSlant.NORMAL
    char = attributes.char

    ctxt.set_font_size(1)
    ctxt.select_font_face(attributes.font, cairo.FONT_SLANT_NORMAL, weight)

    extent = ctxt.text_extents(char)
    font_size = attributes.scale / max(extent.width, extent.height)  # normalize font size
    ctxt.set_font_size(font_size)

    # extent = ctxt.text_extents(char)
    # print(max(extent.width, extent.height))
    # print()
    font_matrix = ctxt.get_font_matrix()

    # set slant to normal and perform it manually. There seems to be some issues with system italic
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

    return extent, None  # TODO verify that the extent is the final extent and not the one before translate


class Pattern(object):
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


class NoPattern(Pattern):
    def draw(self, ctxt):
        pass

    def set_as_source(self, ctxt):
        ctxt.set_source_rgba(1, 1, 1, 0)


class SolidColor(Pattern):
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


class Gradient(Pattern):
    def __init__(self, alpha=1, types=('radial', 'linear'), random_color=None, rng=np.random):
        if random_color is None:
            random_color = color_sampler(rng)
        self.random_color = random_color
        self.rng = rng

        self.types = types
        self.alpha = alpha

    def set_as_source(self, ctxt):
        pat = _random_pattern(self.alpha, self.random_color, rng=self.rng, patern_types=self.types)
        ctxt.set_source(pat)


class MultiGradient(Pattern):
    def __init__(self, alpha=0.5, n_gradients=2, types=('radial', 'linear'), random_color=None, rng=np.random):
        if random_color is None:
            random_color = color_sampler(rng)
        self.random_color = random_color
        self.rng = rng
        self.types = types
        self.alpha = alpha
        self.n_gradients = n_gradients

    def draw(self, ctxt):
        for i in range(self.n_gradients):
            if i == 0:
                alpha = self.alpha
            else:
                alpha = self.alpha
            pat = _random_pattern(alpha, self.random_color, rng=self.rng, patern_types=self.types)
            ctxt.rectangle(0, 0, 1, 1)  # Rectangle(x0, y0, x1, y1)
            ctxt.set_source(pat)
            ctxt.fill()

    def set_as_source(self, ctxt):
        raise NotImplemented()


def _random_pattern(alpha=0.8, random_color=None, patern_types=('linear', 'radial'), rng=np.random):
    """"Select a random pattern with either radioal or linear gradient."""
    if random_color is None:
        random_color = color_sampler(rng)
    pattern_type = rng.choice(patern_types)

    # TODO have the locations dependant on the symbol position, or in general have a more intelligent design
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


class Camouflage(Pattern):
    def __init__(self, stroke_length=0.4, stroke_width=0.05, stroke_angle=np.pi / 4, stroke_noise=0.02, n_stroke=500,
                 rng=np.random):
        self.rng = rng
        self.stroke_length = stroke_length
        self.stroke_width = stroke_width
        self.n_stroke = n_stroke
        self.stroke_angle = stroke_angle
        self.stroke_noise = stroke_noise

    def draw(self, ctxt):
        stroke_vector = self.stroke_length * np.array([np.cos(self.stroke_angle), np.sin(self.stroke_angle)])

        for i in range(self.n_stroke):
            start = (self.rng.rand(2) * 1.6 - 0.3) * (1 - stroke_vector)
            stop = start + stroke_vector + self.rng.randn(2) * self.stroke_noise
            ctxt.move_to(*start)
            ctxt.line_to(*stop)
            ctxt.set_line_width(self.stroke_width)

            b, g, r = self.rng.rand(3)

            ctxt.set_source_rgba(b, g, r, 0.8)
            ctxt.stroke()

    def set_as_source(self, ctxt):
        surface = ctxt.get_group_target()
        width, height = surface.get_width(), surface.get_height()
        source_surface = self.surface(width, height)
        ctxt.set_source_surface(source_surface)

    def to_json(self):
        pass


def _surface_to_array(surface):
    buf = surface.get_data()
    img = np.ndarray(shape=(surface.get_height(), surface.get_width(), 4), dtype=np.uint8, buffer=buf)
    return img[:, :, :3]


def _make_surface(width, height):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    surface.set_device_scale(width, height)
    ctxt = cairo.Context(surface)
    return surface, ctxt


def _image_transform(img, inverse_color, pixel_noise_scale, is_gray, rng):
    img = img.astype(np.float32) / 256.

    if is_gray:
        img = np.mean(img, axis=2, keepdims=True)

    if inverse_color:
        img = 1 - img

    mn, mx = np.min(img), np.max(img)
    img = (img - mn) / (mx - mn)

    img += rng.randn(*img.shape) * pixel_noise_scale
    img = np.clip(img, 0., 1.)

    return (img * 255).astype(np.uint8)


class Image:
    def __init__(self, symbols, resolution=(32, 32), background=NoPattern(), inverse_color=False,
                 pixel_noise_scale=0.01, is_gray=False, rng=np.random):
        self.symbols = symbols
        self.resolution = resolution
        self.inverse_color = inverse_color
        self.pixel_noise_scale = pixel_noise_scale
        self.background = background
        self.is_gray = is_gray
        self.rng = rng

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
        return _image_transform(img, self.inverse_color, self.pixel_noise_scale, self.is_gray, self.rng)

    def attribute_dict(self):
        symbols = [symbol.attribute_dict() for symbol in self.symbols]
        data = dict(
            resolution=self.resolution,
            pixel_noise_scale=self.pixel_noise_scale,
            background=self.background.attribute_dict()
        )
        data.update(symbols[0])  # hack to allow flatten access
        data['symbols'] = symbols

        return data

    def add_symbol(self, symbol):
        self.symbols.append(symbol)


class Symbol:
    """Class containing attributes describing the image

    Attributes:
        alphabet: Object of type Alphabet
        char: string of 1 or more characters in the image
        font: string describing the font used to draw characters
        foreground: object of type Pattern, used for the foreground of the symbol
        is_slant: bool describing if char is italic or not
        is_bold: bool describing if char is bold or not
        rotation: float, rotation angle of the text
        scale: float, scale of the text
        translation: relative (x, y) translation of the text
        rng: random number generator to be used. Defaults to np.random
    """

    def __init__(self, alphabet, char, font, foreground, is_slant, is_bold, rotation, scale, translation,
                 rng=np.random):
        self.alphabet = alphabet
        self.char = char
        self.font = font
        self.is_bold = is_bold
        self.is_slant = is_slant
        self.foreground = foreground
        self.rotation = rotation
        self.scale = scale
        self.translation = translation
        self.rng = rng

    def draw(self, ctxt):
        draw_symbol(ctxt, self)

    def make_mask(self, resolution):
        fg = self.foreground
        self.foreground = SolidColor((1, 1, 1))
        surface, ctxt = _make_surface(*resolution)
        draw_symbol(ctxt, self)
        self.foreground = fg
        img = _surface_to_array(surface)
        return np.mean(img, axis=2, keepdims=True).astype(np.uint8)

    def attribute_dict(self):
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
