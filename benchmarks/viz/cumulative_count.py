import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
from datasets.aleatoric_uncertainty_datasets import AleatoricSynbols
from tqdm import tqdm

FILE = '/home/fred/dump/ckpt_1e098eed78c5044fe470769409260b3e.h5'

kwargs = {
    'path': '/home/fred/dump/missing-symbol_n=100000_2020-Apr-10.h5py',
    'name': 'active_learning',
    'task': 'char',
    'initial_pool': 2000,
    'seed': 1337,
    'uncertainty_config': {'is_bold': {}}}


def main():
    with h5py.File(FILE, 'r') as f:
        acc = []
        for k, v in f.items():
            acc.append(v['labelled'].value)
    n = np.array(acc)
    n_cum = np.cumsum(n, 0)[-1]

    ds = AleatoricSynbols(kwargs, kwargs['path'], split='train', key=kwargs['task'], seed=1337)
    bit = get_aleatoric_bitmap(ds)
    # Train
    start, end = 0, 70000
    aleatoric_idx = np.array(bit)[ds.indices[start:end]]
    toplot = n_cum.max() - n_cum[aleatoric_idx.astype(np.bool)]
    toplot.sort()
    ncum_s = np.copy(n_cum)
    ncum_s.sort()
    ncum_s = ncum_s.max() - ncum_s
    x_idx2, y_count2 = np.unique(ncum_s, return_counts=True)
    x_idx2, y_count2 = x_idx2[1:], y_count2[1:]
    y_count2 = np.cumsum(y_count2)
    x_idx, y_count = np.unique(toplot, return_counts=True)
    y_count = np.cumsum(y_count)

    plt.plot(x_idx, y_count, label='Number of aleatoric sample selected')
    plt.plot(x_idx2, y_count2, label='Dataset size')
    plt.hlines(sum(aleatoric_idx), xmin=0, xmax=max(x_idx2), label='Number of aleatoric sample')
    plt.ylabel('Cumulative count')
    plt.xlabel('Active learning step')
    plt.title('AL Selection for Entropy')
    plt.legend()
    plt.show()


def get_aleatoric_bitmap(ds):
    _, y = ds._load_data(ds.path)
    bit = []
    for yi in tqdm(y):
        d = json.loads(yi)
        d = d['translation']
        if not isinstance(d, list):
            d = [d, d]
        bit.append(1 if any(abs(x) > 1 for x in d) else 0)
    return bit


if __name__ == '__main__':
    main()
