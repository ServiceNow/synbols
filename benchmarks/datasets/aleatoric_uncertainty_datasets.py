import json
from typing import Dict

import numpy as np

from .synbols import Synbols

SLANT_MAP = {
    'italic',
    'normal',
    'oblique'
}


class Attributes:
    def __init__(self, background='gradient', foreground='gradient',
                 slant=None, is_bold=None, rotation=None, scale=None, translation=None,
                 inverse_color=None,
                 pixel_noise_scale=0.01, resolution=(32, 32), rng=np.random.RandomState(42)):
        self.is_bold = rng.choice([True, False]) if is_bold is None else is_bold
        self.slant = rng.choice(list(SLANT_MAP)) if slant is None else slant
        self.background = background
        self.foreground = foreground
        self.rotation = rng.randn() * 0.2 if rotation is None else rotation
        self.scale = tuple(np.exp(rng.randn(2) * 0.1)) if scale is None else scale
        self.translation = tuple(rng.rand(2) * 0.2 - 0.1) if translation is None else translation
        self.inverse_color = rng.choice([True, False]) if inverse_color is None else inverse_color

        self.resolution = resolution
        self.pixel_noise_scale = pixel_noise_scale
        self.rng = rng

        # populated by make_image
        self.text_rectangle = None
        self.main_char_rectangle = None

    def to_dict(self):
        return dict(
            is_bold=str(self.is_bold),
            slant=self.slant,
            scale=self.scale,
            translation=self.translation,
            inverse_color=str(self.inverse_color),
            resolution=self.resolution,
            pixel_noise_scale=self.pixel_noise_scale,
            text_rectangle=self.text_rectangle,
            main_char_rectangle=self.main_char_rectangle,
        )


class AleatoricSynbols(Synbols):
    def __init__(self, uncertainty_config: Dict, path, split, key='font', transform=None, p=0.5,
                 seed=None):
        super().__init__(path=path, split=split, key=key, transform=transform)
        self.uncertainty_config = uncertainty_config
        self.p = p
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        if split == 'train':
            self._create_aleatoric_noise()

    def get_train_values(self, y):
        start = 0
        end = int(0.8 * len(y))
        return y[self.indices[start:end]]

    def _create_aleatoric_noise(self):
        self._latent_space = self._get_latent_space()
        data = np.load(self.path)
        y = data['y']
        del data
        _y = []
        for yi in y:
            j = json.loads(yi)
            _y.append({key: j[key] for key in self.uncertainty_config.keys()})

        y_flag = [self._isin_latent(yi) for yi in _y]
        y_flag = self.get_train_values(np.array(y_flag))
        assert len(y_flag) == len(self.y)
        print(f"{sum(y_flag)} items are in the latent space out of {len(y_flag)}.")
        y_flag = [True if yi and self.rng.rand() < self.p else False for yi in y_flag]
        print(f"{sum(y_flag)} items will be shuffled")
        targets = self.y[y_flag]
        print(f"Classes with some elements shuffled: {np.unique(targets)}")
        self.rng.shuffle(targets)
        self.y[y_flag] = targets

    def __getitem__(self, item):
        return self.transform(self.x[item]), self.y[item]

    def __len__(self):
        return len(self.x)

    def _get_latent_space(self) -> Dict:
        return Attributes(rng=self.rng).to_dict()

    def _isin_latent(self, y):
        """Returns a bitmap of any"""
        flag = True
        for key, val in y.items():
            latent_val = self._latent_space[key]
            if isinstance(val, float):
                scale = self.uncertainty_config[key].get('scale', 0.02)
                flag = flag and (latent_val - scale) < val < (latent_val + scale)
            else:
                flag = flag and (val == latent_val)
        return flag


if __name__ == '__main__':
    synbols = AleatoricSynbols(uncertainty_config={'is_bold': {}},
                               p=0.05,
                               path='/mnt/datasets/public/research/synbols/latin_res=32x32_n=100000.npz',
                               split='train')
