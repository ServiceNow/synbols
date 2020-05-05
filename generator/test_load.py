import numpy as np
from synbols.data_io import load_h5, load_attributes_h5, load_minibatch_h5
import time as t
import h5py

file_path = 'test2.h5py'


# file_path = '../segmentation_n=10000_2020-May-04.h5py'


def write_ds(file_path, n_samples, chunk_size=1):
    with h5py.File(file_path, 'w', libver='latest') as file:
        shape = (32, 32, 3)
        dset = file.create_dataset("x", (n_samples,) + shape, dtype=np.uint8)

        for i in range(n_samples):
            if i % 1000 == 0:
                print('writing %d' % i)
            dset[i] = np.random.randint(0, 2, shape)


def random_indices():
    return np.sort(np.random.choice(5000, 100, replace=False))

    start = np.random.randint(0, 5000)
    return np.arange(start, start + 100)


n_samples = 10000
t0 = t.time()
write_ds(file_path, n_samples)
dt = (t.time() - t0) / n_samples
print('writing: took %.3g ms/images' % (dt * 1000))

for i in range(10):
    indices = random_indices()

    t0 = t.time()
    with h5py.File(file_path, 'r') as fd:
        x = np.array(fd['x'][indices])

    dt = (t.time() - t0) / len(indices)
    print('reading: took %.3g ms/images, x.shape=%s' % ((dt * 1000), x.shape))
