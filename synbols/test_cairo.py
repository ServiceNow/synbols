import cairo
import numpy as np
from matplotlib import pyplot as plt
import time as t
import synbols


def camouflage(ctxt, stroke_length, rng):
    for i in range(50):
        start = rng.rand(2) - stroke_length / 2.
        stop = start + stroke_length + rng.randn(2) * 0.05
        ctxt.move_to(*start)
        ctxt.line_to(*stop)
        ctxt.set_line_width(0.05)

        b, g, r = rng.rand(3)

        ctxt.set_source_rgba(b, g, r, 0.2)
        ctxt.stroke()


def symbol_surface(char, width, height):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctxt = cairo.Context(surface)
    ctxt.scale(width, height)  # Normalizing the canvas

    draw_symbol(ctxt, char)
    return surface


def draw_symbol(ctxt, char='D', font='Gorlock'):
    ctxt.save()
    ctxt.set_font_size(0.7)
    ctxt.select_font_face(font)
    extent = ctxt.text_extents(char)

    ctxt.translate(0.5, 0.5)
    scale = 0.6 / np.maximum(extent.width, extent.height)
    ctxt.scale(scale, scale)

    ctxt.translate(-extent.width / 2., extent.height / 2)

    ctxt.set_source_rgba(1, 0, 0, 0.2)
    ctxt.show_text(char)
    # ctxt.fill()
    ctxt.restore()
    return extent


def make_camouflage(stroke_length=0.4, width=64, height=64, rng=np.random):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctxt = cairo.Context(surface)
    ctxt.scale(width, height)  # Normalizing the canvas

    symbol = symbol_surface('D', width, height)
    ctxt.mask_surface(symbol)
    # draw_symbol(ctxt)
    # ctxt.clip()
    t0 = t.time()
    camouflage(ctxt, stroke_length, rng)
    dt = t.time() - t0
    print('took %.3gs' % dt)
    ctxt.fill()

    buf = surface.get_data()
    img = np.ndarray(shape=(width, height, 4), dtype=np.uint8, buffer=buf)
    plt.imshow(img)
    plt.show()


make_camouflage()
