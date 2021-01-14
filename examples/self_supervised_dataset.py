from synbols.generate import basic_attribute_sampler, rand_seed
from synbols.motion import views_sampler, generate_and_write_dataset
from synbols.fonts import LANGUAGE_MAP
from synbols.drawing import Gradient

alphabet = LANGUAGE_MAP['english'].get_alphabet(support_bold=False, auxiliary=False)

print("Alphabet: %d char, %d fonts" % (len(alphabet.symbols), len(alphabet.fonts)))


def change_position(attr, rng):
    for symbol in attr.symbols:
        symbol.translation += 0.5 * rng.randn(2)
        symbol.rotation += rng.randn()


def change_char(attr, rng):
    for symbol in attr.symbols:
        symbol.char = rng.choice(sorted(alphabet.symbols))


def change_font(attr, rng):
    for symbol in attr.symbols:
        symbol.font = rng.choice(sorted(alphabet.fonts))


def change_bg(attr, rng):
    attr.background = Gradient(seed=rand_seed(rng))


def multi_transform(*transformations):
    def transform(attr, rng):
        for transformation in transformations:
            transformation(attr, rng)

    return transform


n_samples = 100000
resolution = (32, 32)
attr_sampler = basic_attribute_sampler(alphabet=alphabet, resolution=resolution, n_symbols=1, scale=0.7,
                                       max_contrast=False)

trans = [None,
         multi_transform(change_position, change_font),
         multi_transform(change_font, change_bg),
         multi_transform(change_bg,   change_char),
         multi_transform(change_char, change_position)]

sampler = views_sampler(attr_sampler, trans)
file_name = 'synbols_%d_double_transforms_n=%d_res=%dx%d' % (len(trans), n_samples, resolution[0], resolution[1])
generate_and_write_dataset(file_name, sampler, n_samples, preview_shape=(10, 10))

