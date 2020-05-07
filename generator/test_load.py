import numpy as np
from synbols.data_io import load_h5, load_attributes_h5, load_minibatch_h5
import time as t
import h5py

file_path = '../segmentation_n=100000_2020-Apr-30.h5py'


# print('load attributes')
#
# t0 = t.time()
# attr_list, split = load_attributes_h5(file_path)
# print("took %.2fs" % (t0 - t.time()))
#
# print('load dataset')
#
# t0 = t.time()
# x, mask, attr_list, splits = load_h5(file_path)
# print("took %.3fs" % (t0 - t.time()))

def random_indices():

    # return np.sort(np.random.choice(100000, 100, replace=False))
    #
    start = np.random.randint(0, 10000)
    return np.arange(start, start + 100)


for i in range(10):
    indices = random_indices()

    t0 = t.time()
    with h5py.File(file_path, 'r') as fd:
        x = np.array(fd['x'][indices])

    print('took %.3gs for minibatch' % (t.time() - t0))

print(x.shape)

print()
print("open one time")
with h5py.File(file_path, 'r') as fd:
    for i in range(10):

        t0 = t.time()
        indices = random_indices()
        x = np.array(fd['x'][indices])

        print('took %.3gs for minibatch' % (t.time() - t0))
