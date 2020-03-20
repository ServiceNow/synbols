import cairo
import json
import numpy as np

from sys import stdout

from google_fonts import ALPHABET_MAP

SLANT_MAP = {
    cairo.FONT_SLANT_ITALIC: 'italic',
    cairo.FONT_SLANT_NORMAL: 'normal',
    cairo.FONT_SLANT_OBLIQUE: 'oblique',
}


def draw_symbol(ctxt, attributes):
    """Core function drawing the characters as described in `attributes`

    Args:
        ctxt: cairo context to draw the image
        attributes: Object of type Attributes containing information about the image

    Returns:
        extent: rectangle containing the text in the coordinate of the context
        extent_main_char: rectangle containing the central character in the coordinate of the context
    """
    _make_background(ctxt, attributes.background)

    _make_foreground(ctxt, attributes.foreground)

    weight = cairo.FONT_WEIGHT_BOLD if attributes.is_bold else cairo.FONT_WEIGHT_NORMAL

    char = attributes.char

    ctxt.set_font_size(0.7)
    ctxt.select_font_face(attributes.font, attributes.slant, weight)
    extent = ctxt.text_extents(char)

    if len(char) == 3:
        raise NotImplementedError()  # TODO: support multi-part character languages
        extent_main_char = ctxt.text_extents(char[1])
    elif len(char) == 1:
        extent_main_char = extent
    else:
        raise Exception("Unexpected length of string: %d. Should be either 3 or 1" % len(char))

    if extent_main_char.width == 0. or extent_main_char.height == 0:
        stdout.buffer.write(char.encode("utf-8"))
        print("   Font:", attributes.font, "<-- ERROR needs attention")
        return None, None

    ctxt.translate(0.5, 0.5)
    scale = 0.6 / np.maximum(extent_main_char.width, extent_main_char.height)
    ctxt.scale(scale, scale)
    ctxt.scale(*attributes.scale)

    ctxt.rotate(attributes.rotation)

    if len(char) == 3:
        raise NotImplementedError()  # TODO: support multi-part character languages
        ctxt.translate(-ctxt.text_extents(char[0]).x_advance - extent_main_char.width / 2., extent_main_char.height / 2)
        ctxt.translate(*attributes.translation)
    else:
        ctxt.translate(-extent.x_bearing - extent.width / 2, -extent.y_bearing - extent.height / 2)
        ctxt.translate(*attributes.translation)

    ctxt.show_text(char)

    ctxt.clip()
    ctxt.paint()

    return extent, extent_main_char


def _make_foreground(ctxt, style):
    if style == 'gradient':
        pat = random_pattern(0.8, (0.2, 1), patern_types=('linear',))
        ctxt.set_source(pat)
    elif isinstance(style, Camouflage):
        style.set_as_source(ctxt)
    elif style is None:
        ctxt.set_source_rgb(1, 1, 1)
    else:
        raise Exception("Unknown foreground style %s" % style)


def _make_background(ctxt, style, rng=np.random):
    """Random background combining various patterns."""
    if style is None:
        ctxt.set_source_rgb(1, 1, 1)
        ctxt.fill()
    elif isinstance(style, Camouflage):
        style.draw(ctxt)
    elif style == "gradient":
        for i in range(5):
            pat = random_pattern(0.4, (0, 0.8), rng=rng)
            ctxt.rectangle(0, 0, 1, 1)  # Rectangle(x0, y0, x1, y1)
            ctxt.set_source(pat)
            ctxt.fill()
    else:
        raise Exception("Unknown background style %s" % style)


class Camouflage:
    def __init__(self, stroke_length=0.4, stroke_width=0.05, stroke_angle=np.pi / 4, stroke_noise=0.02, n_stroke=500,
                 rng=np.random):
        self.stroke_length = stroke_length
        self.stroke_width = stroke_width
        self.n_stroke = n_stroke
        self.stroke_angle = stroke_angle
        self.stroke_noise = stroke_noise
        self.rng = rng

    def draw(self, ctxt):
        stroke_vector = self.stroke_length * np.array([np.cos(self.stroke_angle), np.sin(self.stroke_angle)])

        for i in range(self.n_stroke):
            start = (self.rng.rand(2) * 1.2 - 0.1) * (1 - stroke_vector)
            stop = start + stroke_vector + self.rng.randn(2) * self.stroke_noise
            ctxt.move_to(*start)
            ctxt.line_to(*stop)
            ctxt.set_line_width(self.stroke_width)

            b, g, r = self.rng.rand(3)

            ctxt.set_source_rgba(b, g, r, 0.8)
            ctxt.stroke()

    def surface(self, width, height):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        surface.set_device_scale(width, height)
        ctxt_bg = cairo.Context(surface)
        self.draw(ctxt_bg)
        return surface

    def set_as_source(self, ctxt):
        surface = ctxt.get_group_target()
        width, height = surface.get_width(), surface.get_height()
        source_surface = self.surface(width, height)
        ctxt.set_source_surface(source_surface)

    def to_json(self):
        pass


class Attributes:
    """Class containing attributes describing the image

    Attributes:
        alphabet: TODO(allac)
        char: string of 1 or more characters in the image
        font: string describing the font used to draw characters
        background: TODO(allac)
        slant: one of cairo.FONT_SLANT_ITALIC, cairo.FONT_SLANT_NORMAL, cairo.FONT_SLANT_OBLIQUE
            default: Uniform random
        is_bold: bool describing if char is bold or not
            default: Uniform random
        rotation: float, rotation angle of the text
            default: Normal(0, 0.2)
        scale: float, scale of the text
            default: Normal(0, 0.1)
        translation: relative (x, y) translation of the text
            default: Normal(0.1, 0.2)
        inverse_color: bool describing if color is inverted or not
            default: Uniform random
        pixel_noise_scale: standard deviation of pixel-wise noise
            default: 0.01
        resolution: tuple of int for (width, height) of the image
            default: (32, 32) (TODO(allac) fix bug when width != height)

    """

    def __init__(self, alphabet=None, char=None, font=None, background='gradient', foreground='gradient',
                 slant=None, is_bold=None, rotation=None, scale=None, translation=None, inverse_color=None,
                 pixel_noise_scale=0.01, resolution=(32, 32), rng=np.random):

        self.alphabet = rng.choice(ALPHABET_MAP.values()) if alphabet is None else alphabet
        self.char = rng.choice(alphabet.symbols) if char is None else char
        self.font = rng.choice(alphabet.fonts) if font is None else font
        self.is_bold = rng.choice([True, False]) if is_bold is None else is_bold
        self.slant = rng.choice(list(SLANT_MAP.keys())) if slant is None else slant
        self.background = background
        self.foreground = foreground
        self.rotation = rng.randn() * 0.2 if rotation is None else rotation
        self.scale = tuple(np.exp(rng.randn(2) * 0.1)) if scale is None else scale
        self.translation = tuple(rng.rand(2) * 0.2 - 0.1) if translation is None else translation
        self.inverse_color = rng.choice([True, False]) if inverse_color is None else inverse_color

        self.resolution = resolution
        self.pixel_noise_scale = pixel_noise_scale
        self.rng = rng

        # populated by make_image
        self.text_rectangle = None
        self.main_char_rectangle = None

    def make_image(self):
        width, height = self.resolution
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctxt = cairo.Context(surface)
        ctxt.scale(width, height)  # Normalizing the canvas
        self.text_rectangle, self.main_char_rectangle = draw_symbol(ctxt, self)
        if self.text_rectangle is None:
            # XXX: for debugging
            from copy import deepcopy
            attr = deepcopy(self)
            attr.font = ""
            attr.char = "#"
            attr.background = None
            print(attr.foreground)
            self.text_rectangle, self.main_char_rectangle = draw_symbol(ctxt, attr)
        buf = surface.get_data()
        img = np.ndarray(shape=(width, height, 4), dtype=np.uint8, buffer=buf)
        img = img.astype(np.float32) / 256.
        img = img[:, :, 0:3]
        if self.inverse_color:
            img = 1 - img

        mn, mx = np.min(img), np.max(img)
        img = (img - mn) / (mx - mn)

        img += self.rng.randn(*img.shape) * self.pixel_noise_scale
        img = np.clip(img, 0., 1.)

        img = (img * 255).astype(np.uint8)
        return img

    def attribute_dict(self):
        return dict(
            alphabet=self.alphabet.name,
            char=self.char,
            font=self.font,
            is_bold=str(self.is_bold),
            slant=SLANT_MAP[self.slant],
            scale=self.scale,
            translation=self.translation,
            inverse_color=str(self.inverse_color),
            resolution=self.resolution,
            pixel_noise_scale=self.pixel_noise_scale,
            text_rectangle=self.text_rectangle,
            main_char_rectangle=self.main_char_rectangle,
        )


def random_pattern(alpha=0.8, brightness_range=(0, 1), patern_types=('linear', 'radial'), rng=np.random):
    """"Select a random pattern with either radioal or linear gradient."""
    pattern_type = rng.choice(patern_types)
    if pattern_type == 'linear':
        y1, y2 = rng.rand(2)
        pat = cairo.LinearGradient(-1, y1, 2, y2)
    if pattern_type == 'radial':
        x1, y1, x2, y2 = rng.randn(4) * 0.5
        pat = cairo.RadialGradient(x1 * 2, y1 * 2, 2, x2, y2, 0.2)

    def random_color():
        b_delta = brightness_range[1] - brightness_range[0]
        return rng.rand(3) * b_delta + brightness_range[0]

    r, g, b = random_color()
    pat.add_color_stop_rgba(1, r, g, b, alpha)
    r, g, b = random_color()
    pat.add_color_stop_rgba(0.5, r, g, b, alpha)
    r, g, b = random_color()
    pat.add_color_stop_rgba(0, r, g, b, alpha)
    return pat


def solid_pattern(alpha=0.8, brightness_range=(0, 1), rng=np.random):
    def random_color():
        b_delta = brightness_range[1] - brightness_range[0]
        return rng.rand(3) * b_delta + brightness_range[0]

    r, g, b = random_color()
    return cairo.SolidPattern(r, g, b, alpha)


def _split(set_, ratios, rng=np.random.RandomState(42)):
    n = len(set_)
    counts = np.round(np.array(ratios) * n).astype(np.int)
    counts[0] = n - np.sum(counts[1:])
    set_ = rng.permutation(set_)
    idx = 0
    sets = []
    for count in counts:
        sets.append(set_[idx:(idx + count)])
        idx += count
    return sets
