import cairo
import numpy as np
import json


from google_fonts import ALPHABET_MAP


def draw_symbol(ctxt, attributes):
    """Core function drawing the characters as described in `attributes`.

    Args:
        ctxt: cairo context to draw the image
        attributes: Object of type Attributes containing information about the
        image

    Returns:
        extent: rectangle containing the text in the coordinate of the context
        extent_main_char: rectangle containing the central character in the
        coordinate of the context
    """
    # Create the background image
    make_background(ctxt, attributes.background)

    # Select if the font is normal or bold
    weight = cairo.FONT_WEIGHT_BOLD if attributes.is_bold else cairo.FONT_WEIGHT_NORMAL

    # Get the character to draw
    char = attributes.char

    # Set the font size, font type, italic and bold attributes. TODO: Fix size?
    ctxt.set_font_size(0.7)
    ctxt.select_font_face(attributes.font, attributes.slant, weight)

    # Control the size. TODO: Why 1 or 3?
    extent = ctxt.text_extents(char)
    if len(char) == 3:
        extent_main_char = ctxt.text_extents(char[1])
    elif len(char) == 1:
        extent_main_char = extent
    else:
        raise Exception("Unexpected length of string: %d. Should be either "
                        "3 or 1" % len(char))
    if extent_main_char.width == 0. or extent_main_char.height == 0:
        # print(char, attributes.font)  # TODO fix printing of unicode in docker
        return None, None

    # Translate, scale and rotate
    ctxt.translate(0.5, 0.5)
    scale = 0.6 / np.maximum(extent_main_char.width, extent_main_char.height)
    ctxt.scale(scale, scale)
    ctxt.scale(*attributes.scale)
    ctxt.rotate(attributes.rotation)

    if len(char) == 3:
        ctxt.translate(-ctxt.text_extents(char[0]).x_advance -
                       extent_main_char.width / 2.,
                       extent_main_char.height / 2)
        ctxt.translate(*attributes.translation)
    else:
        ctxt.translate(-extent.width / 2., extent.height / 2)
        ctxt.translate(*attributes.translation)

    # ??
    if attributes.foreground is not None:
        pat = random_pattern(0.8, (0.2, 1), patern_types=('linear',))
        ctxt.set_source(pat)

    # Draw the symbol
    ctxt.show_text(char)

    return extent, extent_main_char


SLANT_MAP = {
    cairo.FONT_SLANT_ITALIC: 'italic',
    cairo.FONT_SLANT_NORMAL: 'normal',
    cairo.FONT_SLANT_OBLIQUE: 'oblique',
}


class Attributes:
    """Class containing attributes describing the image

    Attributes:
        alphabet: TODO(allac)
        char: string of 1 or more characters in the image
        font: string describing the font used to draw characters
        background: TODO(allac)
        foreground: TODO(allac)
        slant: one of cairo.FONT_SLANT_ITALIC, cairo.FONT_SLANT_NORMAL,
        cairo.FONT_SLANT_OBLIQUE. If None it selects random
            default: Uniform random
        is_bold: bool describing if char is bold or not. If None it selects
        random
            default: Uniform random
        rotation: float, rotation angle of the text. If None it selects random
            default: Normal(0, 0.2)
        scale: float, scale of the text. If None it selects random
            default: Normal(0, 0.1)
        translation: relative (x, y) translation of the text. If None it
        selects random
            default: Normal(0.1, 0.2)
        inverse_color: bool describing if color is inverted or not. If None it
        selects random
            default: Uniform random
        pixel_noise_scale: standard deviation of pixel-wise noise
            default: 0.01
        resolution: tuple of int for (width, height) of the image
            default: (32, 32) (TODO(allac) fix bug when width != height)

    """

    def __init__(self, alphabet=None, char=None, font=None,
                 background='gradient', foreground='gradient', slant=None,
                 is_bold=None, rotation=None, scale=None, translation=None,
                 inverse_color=None, pixel_noise_scale=0.01,
                 resolution=(32, 32), rng=np.random.RandomState(42)):
        # Initialize attributes
        self.alphabet = alphabet
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
        """Create the image defined by the attributes."""
        # Create the empty canvas
        width, height = self.resolution
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctxt = cairo.Context(surface)
        ctxt.scale(width, height)  # Normalizing the canvas

        # Draw the symbol in the canvas
        self.text_rectangle, self.main_char_rectangle = draw_symbol(ctxt, self)
        buf = surface.get_data()
        img = np.ndarray(shape=(width, height, 4), dtype=np.uint8, buffer=buf)
        img = img.astype(np.float32) / 256. #TODO: Divide by 255.????
        img = img[:,:, 0:3]
        
        # Inverse colors if needed
        if self.inverse_color:
            img = 1 - img

        # Normalize
        mn, mx = np.min(img), np.max(img)
        img = (img - mn) / (mx - mn)

        # Add noise
        img += self.rng.randn(*img.shape) * self.pixel_noise_scale
        img = np.clip(img, 0., 1.)

        # Convert to uint8
        img=(img * 255).astype(np.uint8)
    
        return img

    def to_json(self):
        """Save the attributes into a json file."""
        data = dict(
            alphabet=self.alphabet.name,
            char=self.char,
            font=self.font,
            is_bold=str(self.is_bold),
            slant=SLANT_MAP[self.slant],
            #TODO: Background?
            #TODO: Foreground?
            #TODO: Rotation?
            scale=self.scale,
            translation=self.translation,
            inverse_color=str(self.inverse_color),
            resolution=self.resolution,
            pixel_noise_scale=self.pixel_noise_scale,
            text_rectangle=self.text_rectangle,
            main_char_rectangle=self.main_char_rectangle,
        )

        return json.dumps(data)


def random_pattern(alpha=0.8, brightness_range=(0, 1),
                   patern_types=('linear', 'radial'), rng=np.random):
    """"Select a random pattern with either radial or linear gradient."""
    # Select the gradient type
    pattern_type = rng.choice(patern_types)
    if pattern_type == 'linear':
        y1, y2 = rng.rand(2)
        pat = cairo.LinearGradient(-1, y1, 2, y2)
    elif pattern_type == 'radial':
        x1, y1, x2, y2 = rng.randn(4) * 0.5
        pat = cairo.RadialGradient(x1 * 2, y1 * 2, 2, x2, y2, 0.2)
    else:
        raise ValueError("Unknown pattern type " + pattern_type)

    def random_color():
        b_delta = brightness_range[1] - brightness_range[0]
        return rng.rand(3) * b_delta + brightness_range[0]

    # Set the pattern color
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


def make_background(ctxt, style, rng=np.random):
    """Random background combining various patterns."""
    if style is not None:
        for i in range(5):
            pat = random_pattern(0.4, (0, 0.8), rng=rng)
            ctxt.rectangle(0, 0, 1, 1)  # Rectangle(x0, y0, x1, y1)
            ctxt.set_source(pat)
            ctxt.fill()


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


def make_char_grid_from_lang(lang, width, height, char_stride=1, font_stride=1,
                             rng=np.random.RandomState(42)):
    """Temporary high level interface for making a dataset"""
    dataset = []
    # print("building char grid for ")
    for char in lang.symbols[::char_stride]:
        one_class = []
        # print("generating sample for char: %s" % char.encode('utf-8'))  # TODO find how to print this.
        for font in lang.fonts[::font_stride]:
            attributes = Attributes(lang, char, font,
                                    resolution=(width, height), rng=rng)
            x = attributes.make_image()
            one_class.append(x)

        dataset.append(np.stack(one_class))
    return dataset
