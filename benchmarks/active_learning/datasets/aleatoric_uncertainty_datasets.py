import numpy as np
from PIL import Image
from skimage import img_as_float
from skimage.util import random_noise

from datasets.synbols import Synbols


class ColorNoise:
    def __init__(self, p, sigma, seed):
        self.p = p
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)

    def __call__(self, x):
        if self.rng.rand() < self.p:
            x = Image.fromarray((random_noise(img_as_float(x),
                                              var=self.sigma ** 2) * 255).astype(np.uint8))

        return x


def _shuffle_subset(data: np.ndarray, shuffle_prop: float, rng) -> np.ndarray:
    to_shuffle = np.nonzero(rng.rand(data.shape[0]) < shuffle_prop)[0]
    data[to_shuffle, ...] = data[rng.permutation(to_shuffle), ...]
    return data


class AleatoricSynbols(Synbols):
    def __init__(self, path, split, key='font', transform=None, p=0.0,
                 seed=None, n_classes=None, pixel_sigma=0, pixel_p=0):
        super().__init__(path=path, split=split, key=key, transform=transform)
        self.p = p
        self.pixel_sigma = pixel_sigma
        self.pixel_p = pixel_p
        self.seed = seed
        self.noise_classes = n_classes
        self.rng = np.random.RandomState(self.seed)
        if self.pixel_p > 0 and split == 'train':
            # Pixel noise
            self.x = self._add_pixel_noise()
        if self.p > 0:
            # Label noise
            self.y = self._shuffle_label()

    def get_splits(self, source):
        if self.split == 'train':
            start = 0
            end = int(0.7 * len(source))
        elif self.split == 'calib':
            start = int(0.7 * len(source))
            end = int(0.8 * len(source))
        elif self.split == 'val':
            start = int(0.8 * len(source))
            end = int(0.9 * len(source))
        elif self.split == 'test':
            start = int(0.9 * len(source))
            end = len(source)
        return start, end

    def get_values_split(self, y):
        start, end = self.get_splits(source=y)
        return y[self.indices[start:end]]

    def _shuffle_label(self):
        return _shuffle_subset(self.y, self.p, self.rng)

    def _add_pixel_noise(self):
        color_noise = ColorNoise(self.pixel_p, self.pixel_sigma, self.seed)
        return [color_noise(xi) for xi in self.x]

    def __getitem__(self, item):
        return self.transform(self.x[item]), self.y[item]

    def __len__(self):
        return len(self.x)
