"""This script transform a string into a sequence of Synbols images."""

from synbols.data_io import pack_dataset
from synbols import generate
from synbols.utils import make_img_grid
from PIL import Image

text = 'Synbols'
n_trials = 4

for i in range(n_trials):  # try different variants

    print("Trial %d/%d." % (i, n_trials))

    generator = generate.text_generator(text, resolution=(512, 512))

    x, mask, y = pack_dataset(generator)

    img_grid, _, _ = make_img_grid(x, y, None, None, 1, len(text))
    Image.fromarray(img_grid).save('synbols_text_%d.png' % i)
