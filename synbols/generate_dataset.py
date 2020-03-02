import numpy as np
import synbols
import time as t


def make_alphabet_specific_dataset(alphabet, resolution=(32, 32), n_samples=100000, rng=np.random):
    def generator():
        for i in range(n_samples):

            attributes = synbols.Attributes(alphabet, resolution=resolution, rng=rng)
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


if __name__ == "__main__":
    n_samples = 100000
    alphabet = 'latin'
    resolution = (32, 32)
    file_name = '%s_res=%dx%d_n=%d.npz' % (alphabet, resolution[0], resolution[1], n_samples)
    generator = make_alphabet_specific_dataset(synbols.ALPHABET_MAP[alphabet], n_samples=n_samples)
    write_numpy(file_name, generator)
