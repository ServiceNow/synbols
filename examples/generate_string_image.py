"""This script transform a string into a sequence of Synbols images."""

from synbols.data_io import pack_dataset
from synbols import generate
from synbols.utils import make_img_grid
from PIL import Image

text = 'Synbols'
n_trials = 4

for seed in range(n_trials):  # try different variants

    print("Seed %d/%d." % (seed, n_trials))

    generator = generate.text_generator(text, seed=seed, resolution=(512, 512))

    x, mask, y = pack_dataset(generator)

    img_grid, _, _ = make_img_grid(x, y, None, None, 1, len(text))
    Image.fromarray(img_grid).save('synbols_text_seed:%d.png' % seed)
