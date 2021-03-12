from synbols.generate import generate_and_write_dataset, basic_attribute_sampler
from synbols.drawing import SolidColor
from synbols.generate import LANGUAGE_MAP
import numpy as np

n_samples = 100


def random_solid(rng):
    return SolidColor(color=rng.rand(3))


alphabet = LANGUAGE_MAP['english'].get_alphabet(support_bold=True, auxiliary=False)

# select a fixed random subset of 100 fonts
rng = np.random.RandomState(43)
alphabet.fonts = rng.choice(alphabet.fonts, 100)

attr_sampler = basic_attribute_sampler(alphabet=alphabet, background=random_solid, foreground=random_solid,
                                       pixel_noise_scale=0., max_contrast=False)

file_name = 'synbols_semantic_explanation_n=%d' % (n_samples)
generate_and_write_dataset(file_name, attr_sampler, n_samples)
