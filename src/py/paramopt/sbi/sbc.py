#!/usr/bin/env python3
'''
Routines to perform simulation based calibration (SBC).
'''
from functools import partial
from typing import List, Optional
import pandas as pd
import numpy as np

from sbi.inference.posteriors.base_posterior import NeuralPosterior


def sbc_data(posterior: NeuralPosterior,
             observations: np.ndarray,
             num_sbc_samples: Optional[int] = None,
             num_samples: int = 1000,
             parameter_names: Optional[List[str]] = None) -> pd.DataFrame:
    '''
    Condition the posterior on each observation and draw posterior samples.

    :param posterior: Posterior from which samples are drawn.
    :param observations: Observations for which to draw posterior samples.
    :param num_sbc_samples: Number of samples/observations for which sbc
            samples are drawn. If not provided, all observations in the given
            data frame are considered.
    :param num_samples: Number of samples to draw for each observation.
    :param parameter_names: Names of the parameters drawn from the posterior
        distribution. If None, generic names are used.

    :return: Data frame with the index of the observation in one column
        and the data of the samples in the other column. Each row represents
        one sample.
    '''
    # observation might be one-dimensional -> add extra dimension to make
    # code below more generic
    if observations.ndim == 1:
        observations = observations[:, None]
    idx_with_observations = np.all(~np.isnan(observations),
                                   axis=1).nonzero()[0]

    if num_sbc_samples is not None and \
            num_sbc_samples < len(idx_with_observations):
        idx_with_observations = np.random.choice(idx_with_observations,
                                                 size=num_sbc_samples,
                                                 replace=False)

    results = []
    for obs_id in idx_with_observations:
        results.append(
            posterior.sample((num_samples,), x=observations[obs_id],
                             show_progress_bars=False).cpu().numpy())
    results = np.array(results)

    if parameter_names is None:
        parameter_names = [f"P{i}" for i in range(results.shape[-1])]
    data = pd.DataFrame(results.reshape([-1, results.shape[-1]]),
                        columns=pd.MultiIndex.from_product([['parameters'],
                                                            parameter_names]))
    data['obs_id'] = idx_with_observations.repeat(num_samples)
    return data


def rank(item_to_rank: np.ndarray, samples: np.ndarray) -> np.ndarray:
    '''
    Calculate the rank of the given item in the samples.

    :param item_to_rank: Item (multi-dimensional vector) which should be
        ranked.
    :param samples: Array with different samples in the first dimension and the
        values in the second dimension, i.e., the second dimension should have
        the same length as `item_to_rank).
    '''
    return (samples < item_to_rank).sum(0)


def expected_coverage(original_parameters: np.ndarray,
                      observations: np.ndarray,
                      samples: np.ndarray,
                      posterior: NeuralPosterior, *,
                      alphas: np.ndarray) -> np.ndarray:
    '''
    Calculate the expected coverage.

    Based on Hermans et al. (2022). "A Crisis In Simulation-Based Inference?
    Beware, Your Posterior Approximations Can Be Unfaithful". In: Transactions
    on Machine Learning Research.

    :param original_parameters: Parameters with which the observations were
        created.
    :param observations: Observations of original parameters. With these
        observations samples from the posterior were drawn.
    :param samples: Posterior samples for the given observations. This array
        has the shape (num_observations, num_pos_samples, n_dim_parameter).
        Where num_pos_samples is the number of samples drawn from each
        posterior.
    :param posterior: Posterior for which to determine the expected coverage.
    :param alphas: Alpha at which the expected coverage is calculated.
    :return: Expected coverage for 1 - alpha.
    '''
    log_unnormed = partial(posterior.log_prob, norm_posterior=False)

    ranks = []
    for parameter, obs, pos_samples in zip(original_parameters,
                                           observations,
                                           samples):
        log_prob_original = log_unnormed(parameter, x=obs).numpy()
        log_prob_samples = log_unnormed(pos_samples, x=obs).numpy()
        ranks.append(rank(log_prob_original, log_prob_samples))

    norm_ranks = np.array(ranks) / samples.shape[1]
    in_conv_region = norm_ranks[:, None] > alphas

    return in_conv_region.sum(0) / original_parameters.shape[0]
