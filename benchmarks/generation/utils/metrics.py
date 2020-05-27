import os
import logging
import math
from functools import reduce
from collections import defaultdict
import json
from timeit import default_timer

from tqdm import trange, tqdm
import numpy as np
import torch
import sklearn.metrics
import sklearn.svm as svm
import multiprocessing
import time

def generator(mus, mus_test, ys, ys_test, num_latents, num_factors):
    for i in range(num_latents):
        for j in range(num_factors):
            mu_i = mus[i, :]
            y_j = ys[j, :]
            mu_i_test = mus_test[i, :]
            y_j_test = ys_test[j, :]
            yield mu_i, mu_i_test, y_j, y_j_test
def process_latent(arg):
    mu_i, mu_i_test, y_j, y_j_test = arg
    # Attribute is considered discrete.
    classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
    classifier.fit(mu_i[:, np.newaxis], y_j)
    pred = classifier.predict(mu_i_test[:, np.newaxis])
    return np.mean(pred == y_j_test)

def compute_mig(mus_train, ys_train):
    from lib.disentanglement_lib.disentanglement_lib.evaluation import evaluate
    import lib.disentanglement_lib.disentanglement_lib.evaluation.metrics as metrics
    import gin
    gin_bindings = [
        # "evaluation.evaluation_fn = @mig",
        "dataset.name='auto'",
        "evaluation.random_seed = 0",
        "mig.num_train=1000",
        "discretizer.discretizer_fn = @histogram_discretizer",
        "discretizer.num_bins = 20"
    ]
    gin.parse_config_files_and_bindings([], gin_bindings)
    return metrics.mig._compute_mig(mus_train, ys_train)


def compute_sap(mus, ys):
    var = np.var(mus, 0)
    limit = np.quantile(var, 0.05)
    mus = mus[:, var > limit]
    mus = mus[:, :, None].repeat(ys.shape[1], 2)
    ys = ys[:, None, :].repeat(mus.shape[1], 1)
    c1 = (mus * (1 - ys)).sum(0, keepdims=True) / \
        (1 - ys).sum(0, keepdims=True)
    c2 = (mus * ys).mean(0, keepdims=True) / ys.sum(0, keepdims=True)
    d1 = np.abs(mus - c1)
    d2 = np.abs(mus - c2)
    score = ((d1 > d2).astype(float) == ys).astype(float).mean(0)
    ret = 0
    for factor in range(mus.shape[2]):
        sscore = np.sort(score[:, factor])
        try:
            ret += sscore[-1] - sscore[-2]
        except:
            return -1
    return ret / mus.shape[2]


def _compute_sap(mus, ys, continuous_factors=False):
    """Computes score based on both training and testing codes and factors."""
    mus = mus.transpose()
    ys = ys[:, None].transpose()
    mus_test = mus[:, -5000:]
    ys_test = ys[:, -5000:]
    mus = mus[:, :10000]
    ys = ys[:, :10000]
    score_matrix = compute_score_matrix(mus, ys, mus_test,
                                        ys_test, continuous_factors)
    # Score matrix should have shape [num_latents, num_factors].
    assert score_matrix.shape[0] == mus.shape[0]
    assert score_matrix.shape[1] == ys.shape[0]
    return compute_avg_diff_top_two(score_matrix)


def compute_score_matrix(mus, ys, mus_test, ys_test, continuous_factors):
    """Compute score matrix as described in Section 3."""
    num_latents = mus.shape[0]
    num_factors = ys.shape[0]
    if False:
        score_matrix = np.zeros([num_latents, num_factors])
        for i in range(num_latents):
            for j in range(num_factors):
                mu_i = mus[i, :]
                y_j = ys[j, :]
                if continuous_factors:
                    # Attribute is considered continuous.
                    cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
                    cov_mu_y = cov_mu_i_y_j[0, 1]**2
                    var_mu = cov_mu_i_y_j[0, 0]
                    var_y = cov_mu_i_y_j[1, 1]
                    if var_mu > 1e-12:
                        score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
                    else:
                        score_matrix[i, j] = 0.
                else:
                    # Attribute is considered discrete.
                    mu_i_test = mus_test[i, :]
                    y_j_test = ys_test[j, :]
                    classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                    classifier.fit(mu_i[:, np.newaxis], y_j)
                    pred = classifier.predict(mu_i_test[:, np.newaxis])
                    score_matrix[i, j] = np.mean(pred == y_j_test)
    else:
        with multiprocessing.Pool(16) as pool:
            score_matrix = np.array(pool.map(process_latent, generator(mus, mus_test, ys, ys_test, num_latents, num_factors)))
    return score_matrix.reshape((num_latents, num_factors))


def compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])
