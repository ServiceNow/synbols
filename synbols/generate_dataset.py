#!/usr/bin/python


import synbols
import time as t
import cairo
import argparse
from data_io import write_jpg_zip
import logging
import numpy as np
import synbols

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='name of the predefined dataset', default='default')
parser.add_argument('--n_samples', help='number of samples to generate', type=int, default=100000)


def attribute_generator(n_samples, n_synbols_per_image, **kwargs):
    """Generic attribute generator. kwargs is directly passed to the Attributes constructor."""
    for i in range(n_samples):
        if n_synbols_per_image == 1:
            yield synbols.Attributes(**kwargs)
        else:
            yield [synbols.Attributes(**kwargs) for j in range(n_synbols_per_image)]


def create_cairo_surface_and_ctxt(resolution):
    width, height = resolution
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctxt = cairo.Context(surface)
    ctxt.scale(width, height)  # Normalizing the canvas
    return surface, ctxt


def draw_synbol(attributes, ctxt):
    text_rectangle, main_char_rectangle = synbols.draw_symbol(ctxt, attributes)
    return text_rectangle, main_char_rectangle


def cairo_surface_to_image(surface, resolution, rng, inverse_color, pixel_noise_scale):
    # Cairo to img
    buf = surface.get_data()
    width, height = resolution
    img = np.ndarray(shape=(width, height, 4), dtype=np.uint8, buffer=buf)
    img = img.astype(np.float32) / 256.
    img = img[:, :, 0:3]
    
    # Invers color
    if inverse_color:
        img = 1 - img

    # Normalize image
    mn, mx = np.min(img), np.max(img)
    img = (img - mn) / (mx - mn)

    # Add noise
    img += rng.randn(*img.shape) * pixel_noise_scale
    img = np.clip(img, 0., 1.)

    # Convert to uint8
    img=(img * 255).astype(np.uint8)

    return img


def cairo_surface_to_GT(surface, resolution, rng, inverse_color, pixel_noise_scale):
    # Cairo to img
    buf = surface.get_data()
    width, height = resolution
    img = np.ndarray(shape=(width, height, 4), dtype=np.uint8, buffer=buf)
    img = img[:, :, 0:3]

    return img



def dataset_generator(attr_generator, n_samples, n_synbols_per_image=1):
    """High level function generating the dataset from an attribute generator."""
    t0 = t.time()
    for i, attributes in enumerate(attr_generator):

        if n_synbols_per_image==1:
            # x = attributes.make_image()
            # y = attributes.attribute_dict()
            surface, ctxt = create_cairo_surface_and_ctxt(attributes.resolution)
            synbols._make_background(ctxt, attributes.background)
            attributes.draw_synbol(ctxt, False)
            x = cairo_surface_to_image(surface, attributes.resolution, attributes.rng, attributes.inverse_color, attributes.pixel_noise_scale)
            x2 = None
            y = attributes.attribute_dict()
        else:
            # Create the background image
            surface, ctxt = create_cairo_surface_and_ctxt(attributes[0].resolution)
            synbols._make_background(ctxt, attributes[0].background)

            # # Create the segmentation mask image with zeros
            # surfaceGT, ctxtGT = create_cairo_surface_and_ctxt(attributes[0].resolution)
            # synbols._make_background(ctxtGT, None)
            # synbols._make_foreground(ctxtGT, None)

            y = []
            x2 = np.zeros((attributes[0].resolution), dtype=np.uint8)
            # Draw each character
            for i, attrib in enumerate(attributes):
                # Draw synbol
                ctxt.save()
                attrib.draw_synbol(ctxt, select_background=False, select_foreground=True)
                ctxt.restore()

                # Create the segmentation mask image
                surfaceGT, ctxtGT = create_cairo_surface_and_ctxt(attributes[0].resolution)
                synbols._make_background(ctxtGT, None)
                synbols._make_foreground(ctxtGT, None)
                attrib.draw_synbol(ctxtGT, select_background=False, select_foreground=False)
                xGT = cairo_surface_to_GT(surfaceGT, attributes[0].resolution, attributes[0].rng, attributes[0].inverse_color, attributes[0].pixel_noise_scale)
                xGT_mask = np.sum(xGT, axis=2) > 0
                x2[xGT_mask] = (i+1)*255/6.
                y.append(attrib.attribute_dict())

            # Convert cairo surface into an image
            x = cairo_surface_to_image(surface, attributes[0].resolution, attributes[0].rng, attributes[0].inverse_color, attributes[0].pixel_noise_scale)

        if i % 100 == 0 and i != 0:
            dt = (t.time() - t0) / 100.
            eta = (n_samples - i) * dt
            eta_str = t.strftime("%Hh%Mm%Ss", t.gmtime(eta))

            logging.info("generating sample %4d / %d (%.3g s/image) ETA: %s", i, n_samples, dt, eta_str)
            t0 = t.time()
        yield x, x2, y


def generate_char_grid(alphabet_name, n_char, n_font, rng=np.random, **kwargs):
    def _attr_generator():
        alphabet = synbols.ALPHABET_MAP[alphabet_name]

        chars = rng.choice(alphabet.symbols, n_char, replace=False)
        fonts = rng.choice(alphabet.fonts, n_font, replace=False)

        for char in chars:
            for font in fonts:
                yield synbols.Attributes(alphabet, char, font, rng=rng, **kwargs)

    return dataset_generator(_attr_generator(), n_char * n_font)


def generate_plain_dataset(n_samples, n_synbols_per_image=1):
    alphabet = synbols.ALPHABET_MAP['latin']
    attr_generator = attribute_generator(n_samples, n_synbols_per_image, alphabet=alphabet, background=None, foreground=None,
                                         slant=cairo.FontSlant.NORMAL, is_bold=False, rotation=0, scale=(1., 1.),
                                         translation=(0., 0.), inverse_color=False, pixel_noise_scale=0.)
    return dataset_generator(attr_generator, n_samples, n_synbols_per_image)


def generate_default_dataset(n_samples, n_synbols_per_image=1):
    alphabet = synbols.ALPHABET_MAP['latin']
    attr_generator = attribute_generator(n_samples, n_synbols_per_image, alphabet=alphabet, slant=cairo.FontSlant.NORMAL, is_bold=False)
    return dataset_generator(attr_generator, n_samples, n_synbols_per_image)


def generate_camouflage_dataset(n_samples, n_synbols_per_image=1):
    alphabet = synbols.ALPHABET_MAP['latin']
    fg = synbols.Camouflage(stroke_angle=0.5)
    bg = synbols.Camouflage(stroke_angle=1.)
    attr_generator = attribute_generator(n_samples, n_synbols_per_image, alphabet=alphabet, is_bold=True, foreground=fg, background=bg,
                                         scale=(1.3, 1.3))
    return dataset_generator(attr_generator, n_samples, n_synbols_per_image)


def generate_segmentation_dataset(n_samples, n_synbols_per_image=5):
    alphabet = synbols.ALPHABET_MAP['latin']
    attr_generator = attribute_generator(n_samples, n_synbols_per_image, alphabet=alphabet, slant=cairo.FontSlant.NORMAL,
        is_bold=False, resolution=(128, 128), background='gradient', n_symbols_per_image=2, inverse_color=False)
    return dataset_generator(attr_generator, n_samples, n_synbols_per_image)


DATASET_GENERATOR_MAP = {
    'plain': generate_plain_dataset,
    'default': generate_default_dataset,
    'camouflage': generate_camouflage_dataset,
    'segmentation': generate_segmentation_dataset,
}

if __name__ == "__main__":
    args = parser.parse_args()

    logging.info("Generating %d samples from %s dataset", args.n_samples, args.dataset)
    ds_generator = DATASET_GENERATOR_MAP[args.dataset](args.n_samples)

    directory = '%s_n=%d' % (args.dataset, args.n_samples)
    write_jpg_zip(directory, ds_generator)
