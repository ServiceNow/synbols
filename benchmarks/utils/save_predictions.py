import json
import pickle
from pprint import pprint

import numpy as np
import torch
from baal.active.heuristics import requireprobs, AbstractHeuristic, xlogy
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from models import ActiveLearning

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

kwargs = {
    'path': '/mnt/datasets/public/research/synbols/missing-symbol_n=100000_2020-Apr-10.h5py',
    'name': 'active_learning',
    'task': 'char',
    'initial_pool': 2000,
    'seed': 1337,
    'uncertainty_config': {'is_bold': {}}}

exp_dict = {
    'lr': 0.001,
    'batch_size': 32,
    'model': "active_learning",
    'seed': 1337,
    'mu': 1e-3,
    'reg_factor': 1e-4,
    'backbone': "vgg16",
    'num_classes': 52,
    'query_size': 100,
    'shuffle_prop': 0.0,
    'learning_epoch': 10,
    'heuristic': 'bald',
    'iterations': 20,
    'max_epoch': 2000,
    'imagenet_pretraining': True,
    'dataset': kwargs
}


class LeftRightBALD(AbstractHeuristic):
    """
    Sort by the highest acquisition function value.

    Args:
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results
            (default: 'none`).

    References:
        https://arxiv.org/abs/1703.02910
    """

    def __init__(self, shuffle_prop=0.0, reduction='none'):
        super().__init__(
            shuffle_prop=shuffle_prop, reverse=True, reduction=reduction
        )

    @requireprobs
    def compute_score(self, predictions):
        """
        Compute the score according to the heuristic.

        Args:
            predictions (ndarray): Array of predictions

        Returns:
            Array of scores.
        """
        assert predictions.ndim >= 3
        # [n_sample, n_class, ..., n_iterations]

        expected_entropy = - np.mean(np.sum(xlogy(predictions, predictions), axis=1),
                                     axis=-1)  # [batch size, ...]
        expected_p = np.mean(predictions, axis=-1)  # [batch_size, n_classes, ...]
        entropy_expected_p = - np.sum(xlogy(expected_p, expected_p),
                                      axis=1)  # [batch size, ...]
        return entropy_expected_p, expected_entropy


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


def main():
    model = ActiveLearning(exp_dict)
    warmup = 3
    al_dataset = get_dataset(split='train', dataset_dict=kwargs)
    val = get_dataset(split='val', dataset_dict=kwargs)
    ds = al_dataset._dataset
    bit = get_aleatoric_bitmap(ds)

    # Train
    start, end = 0, 70000
    aleatoric_idx = np.array(bit)[ds.indices[start:end]].astype(np.bool)
    for w in range(warmup):
        model.train_on_loader(DataLoader(al_dataset, batch_size=32, num_workers=0))
        model.val_on_loader(DataLoader(val, batch_size=32, num_workers=0))

        pool_pred = model.wrapper.predict_on_dataset(al_dataset._dataset, 32,
                                                     iterations=exp_dict['iterations'],
                                                     use_cuda=True, workers=4)
        left, right = LeftRightBALD().compute_score(pool_pred)
        d = {
            'aleatoric_pred': pool_pred[aleatoric_idx],
            'norma_pred': pool_pred[~aleatoric_idx],
            'all_pred': pool_pred,
            'aleatoric_left': left[aleatoric_idx],
            'normal_left': left[~aleatoric_idx],
            'left': left,
            'aleatoric_right': right[aleatoric_idx],
            'normal_right': right[~aleatoric_idx],
            'right': right,
            'aleatoric_idx': aleatoric_idx
        }
        pprint({k: np.mean(v) for k, v in d.items() if 'pred' not in k})
        print(f"Aleatoric BALD:", np.mean(d['aleatoric_left'] - d['aleatoric_right']))
        print(f"Normal BALD:", np.mean(d['normal_left'] - d['normal_right']))
        pickle.dump(d, open(f'/mnt/projects/bayesian-active-learning/synbols_ckpt/preds_{w}.pkl',
                            'wb'))


if __name__ == '__main__':
    main()
