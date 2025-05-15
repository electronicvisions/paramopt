'''
Simulation-based Inference

Collect functions which are related to perform simulation-based inference
for experiments. The algorithms include sequential neural estimator algorithms
(SNxE) as well as  Monte-Carlo ABC (MCABC).
'''
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch

from sbi.inference import SNPE, SNRE, MCABC
from sbi.inference.posteriors.base_posterior import NeuralPosterior
from sbi.utils.user_input_checks import prepare_for_sbi
from sbi.utils.simulation_utils import simulate_for_sbi
from sbi.utils import get_density_thresholder, RestrictedPrior


class Algorithm(Enum):
    SNPE = auto()
    # SNLE = auto() Not supported due to error during training
    SNRE = auto()
    MCABC = auto()


algorithms = {Algorithm.SNPE: SNPE,
              # Algorithm.SNLE: SNLE,
              Algorithm.SNRE: SNRE,
              Algorithm.MCABC: MCABC}


class SequentialEstimation:
    '''
    Perform sequential estimation.

    :ivar inference: Inference object.
    :ivar train_arguments: Arguments used during the training of
        the inference object. Should be a dictionary with the
        argument name and the value.
    '''
    def __init__(self,
                 algorithm: Algorithm,
                 prior: torch.distributions.Distribution,
                 simulator: Callable, *,
                 target: torch.Tensor,
                 density_estimator: Optional[Union[str, Callable]] = None):
        '''
        :param algorithm: Sequential algorithm to use.
        :param prior: Prior distribution.
        :param simulator: Simulator mapping parameters to results.
        :param target: Target observation.
        :param density_estimator: Density estimator network to use as an
            estimaton for the posterior.
        '''
        self._algorithm = algorithm
        self._prior = prior
        self._target = target
        self._truncated_proposal = False
        self._quantile = 1e-4
        self._proposal_samples = 100_000

        # Prepare inference; start with prior as proposal
        self._simulator, self._proposal = prepare_for_sbi(simulator, prior)

        if algorithm == Algorithm.SNRE:
            self.inference = algorithms[algorithm](
                prior=prior, device=str(prior.variance.device))
        else:
            self.inference = \
                algorithms[algorithm](prior=prior,
                                      density_estimator=density_estimator,
                                      device=str(prior.variance.device))
        self.train_arguments = {}

    def use_truncated_proposal(
            self, quantile: Optional[float] = None,
            num_samples_to_estimate_support: Optional[int] = None) -> None:
        '''
        Use truncated proposals.

        Draw samples from the prior and reject if not in support of estimated
        posterior.

        :param quantile: Samples within the `1-quantile` high-probability
            region of the posterior are accepted.
        :param num_samples_to_estimate_support: Number of samples drawn from
            the approximated posterior to estimate its support.
        '''
        self._truncated_proposal = True
        if quantile is not None:
            self._quantile = quantile
        if num_samples_to_estimate_support is not None:
            self._proposal_samples = num_samples_to_estimate_support

    def next_round(self, n_sim: int = 1000
                   ) -> Tuple[NeuralPosterior, torch.Tensor, torch.Tensor]:
        '''
        Perform next approximation round.

        :param n_sim: Number of simulations to run.
        :return: Approximated posterior, drawn samples, observations.
        '''
        theta, obs = simulate_for_sbi(self._simulator,
                                      self._proposal,
                                      num_simulations=n_sim)
        posterior = self.add_simulations(theta=theta, obs=obs)
        return posterior, theta, obs

    def add_simulations(self,
                        theta: torch.Tensor,
                        obs: torch.Tensor) -> NeuralPosterior:
        '''
        Add the given observations to the inference and update the
        posterior.

        In order for this to work, the parameters have to be sampled
        from the proposal. This is the prior in the first round and
        the last approximation of the posterior in all subsequent
        rounds.

        :param theta: Parameters with which the observations were
            generated. They have to be in the same order as returned
            by the proposal/prior.
        :param obs: Observations to append to the inference.
        :return: Approximated posterior.
        '''

        kwargs = {'proposal': self._proposal} if \
            self._algorithm == Algorithm.SNPE else {}
        self.inference.append_simulations(theta, obs, **kwargs)


        kwargs = {'force_first_round_loss': True} if \
            self._truncated_proposal else {}
        kwargs.update(self.train_arguments)
        self.inference.train(**kwargs)

        posterior = self.inference.build_posterior()

        if self._truncated_proposal:
            accept_reject_fn = get_density_thresholder(
                posterior.set_default_x(self._target),
                quantile=self._quantile,
                num_samples_to_estimate_support=self._proposal_samples)
            self._proposal = RestrictedPrior(self._prior, accept_reject_fn,
                                             sample_with="rejection")
        else:
            self._proposal = posterior.set_default_x(self._target)

        return posterior


def perform_sequential_estimation(
    algorithm: Algorithm,
    proposal: torch.distributions.Distribution,
    simulator: Callable,
    target: np.ndarray, *,
    simulations: Optional[List[int]] = None,
    density_estimator: Optional[Union[str, Callable]] = None,
) -> Tuple[np.ndarray, List]:
    '''
    Perform a sequential neural posterior estimation.

    :param algorithm: Sequential algorithm to use.
    :param proposal: Proposal distribution for the first round of the
        algorithm. This is typically the prior distribution.
    :param simulator: Simulation functions which yields the observation given
        the parameters as an input.
    :param target: Target observation used to condition the posterior on
        starting from the second round.
    :param simulations: List of simulations in each round.
    :param density_estimator: Neural density estimator for the SNPE algorithm.
    :returns: Tuple of array with sample information and List of posteriors for
        each round. The array contains the proposal samples drawn in each
        round, the observations (if `log_observations` is True) and the
        corresponding round.
    '''
    if simulations is None:
        simulations = [1000] * 10

    seq_estimator = SequentialEstimation(algorithm, proposal, simulator,
                                         target=target,
                                         density_estimator=density_estimator)

    logs = []
    posteriors = []
    # Perform inference
    for curr_round, n_simulations in enumerate(simulations):
        posterior, theta, obs = seq_estimator.next_round(n_simulations)

        logs.append(
            np.hstack([theta, obs, np.full((len(obs), 1), curr_round)]))
        posteriors.append(posterior)

    return np.concatenate(logs), posteriors


def perform_mcabc(
    prior: torch.distributions.Distribution,
    simulator: Callable,
    target: np.ndarray,
    eps: float,
    simulations: int = 10000,
) -> np.ndarray:
    '''
    Perform Monte-Carlo ABC.

    :param prior: Probability distribution from which model parameters are
        drawn.
    :param simulator: Simulation functions which yields the observation given
        the parameters as an input.
    :param target: Target observation to which simulation results are compared.
    :param eps: Epsilon for which parameters are accepted.
    :param simulations: Number of simulations to perform.
    :returns: Array which contains the proposal samples drawn in each
        round and the observations.
    '''
    # MCABC wants a batch dimension
    def batch_simulator(params):
        return torch.Tensor(simulator(params[0]))[None, :]

    mcabc = MCABC(batch_simulator, prior)
    theta, summary = mcabc(torch.tensor(target),
                           simulations,
                           eps=eps,
                           return_summary=True)

    return np.hstack([theta, summary['x']])
