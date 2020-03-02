import matplotlib.pyplot as plt
import numpy as np
import synbols
import google_fonts


def plot_dataset(ds, name="dataset", class_stride=1, save_path="."):
    fig = plt.figure(name)
    plt.axis('off')

    img_grid = []
    for one_class in ds[::class_stride]:
        img_grid.append(np.hstack([x for x in one_class]))
    img_grid = np.vstack(img_grid)
    plt.imshow(img_grid)

    fig.tight_layout()
    plt.savefig("%s/%s.png" % (save_path, name), dpi=600)


if __name__ == "__main__":

    print("Alphabets:")
    for alphabet_name, alphabet in synbols.ALPHABET_MAP.items():
        print("%s : %d fonts" % (alphabet_name, len(alphabet.fonts)))

    dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['latin'], 64, 64, 1, 20)  # TODO missing upper cases
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['telugu'], 64, 64, 1, 1)  # TODO some chars are really off centered and out of cells
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['thai'], 64, 64, 1, 1)   # TODO some chars are really off centered and out of cells
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['vietnamese'], 64, 64, 1, 4)  # mostly latin with accents
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['arabic'], 64, 64, 1, 1)
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['hebrew'], 64, 64, 1, 1)
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['khmer'], 64, 64, 3, 1)
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['tamil'], 64, 64, 3, 1)
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['gujarati'], 64, 64, 3, 1)
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['bengali'], 64, 64, 3, 1)
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['malayalam'], 64, 64, 3, 1)
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['korean'], 64, 64, 10, 1)  # TODO: empty cells
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['chinese-simplified'], 64, 64, 10, 1)
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['greek'], 64, 64, 3, 2)
    # dataset = synbols.make_char_grid_from_lang(synbols.ALPHABET_MAP['cyrillic'], 64, 64, 2, 4)

    plot_dataset(dataset)
