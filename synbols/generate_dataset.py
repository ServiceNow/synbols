#!/usr/bin/python


import numpy as np
import synbols
import time as t
import cairo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='name of the predefined dataset', default='plain')
parser.add_argument('--n_samples', help='number of samples to generate', type=int, default=10000)


def alphabet_specific_generator(alphabet, resolution=(32, 32), n_samples=100000, rng=np.random):
    def generator():
        for i in range(n_samples):
            yield synbols.Attributes(alphabet, resolution=resolution, rng=rng)

    return generator


def attribute_generator(n_samples, **kwargs):
    def generator():
        for i in range(n_samples):
            yield synbols.Attributes(**kwargs)

    return generator


def dataset_generator(attr_generator):
    def generator():
        for i, attributes in enumerate(attr_generator()):

            t0 = t.time()
            x = attributes.make_image()
            dt = t.time() - t0
            y = attributes.to_json()

            if i % 100 == 0:
                print("generating sample %d (%.3gs / image)" % (i, dt))

            yield x, y

    return generator


def write_numpy(file_path, generator):
    x, y = zip(*list(generator()))
    x = np.stack(x)

    print("Saving dataset in %s." % file_path)
    np.savez(file_path, x=x, y=y)


def generate_plain_dataset(n_samples):
    alphabet = synbols.ALPHABET_MAP['latin']
    attr_generator = attribute_generator(n_samples, alphabet=alphabet, background=None, foreground=None,
                                         slant=cairo.FontSlant.NORMAL, is_bold=False, rotation=0, scale=(1., 1.),
                                         translation=(0., 0.), inverse_color=False, pixel_noise_scale=0.)
    return dataset_generator(attr_generator)


DATASET_GENERATOR_MAP = {
    'plain': generate_plain_dataset,
}

if __name__ == "__main__":
    args = parser.parse_args()
    ds_generator = DATASET_GENERATOR_MAP[args.dataset](args.n_samples)

    file_name = '%s_n=%d.npz' % (args.dataset, args.n_samples)
    write_numpy(file_name, ds_generator)
