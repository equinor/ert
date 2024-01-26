import numpy as np
import pytest
from iterative_ensemble_smoother.experimental import (
    ensemble_smoother_update_step_row_scaling,
)

from ert.analysis.row_scaling import RowScaling


# We fix the random seed in the tests to ensure no flakiness
@pytest.fixture(autouse=True)
def fix_seed(seed=321):
    np.random.seed(seed)


# The following tests follow the
# posterior properties described in
# https://ert.readthedocs.io/en/latest/theory/ensemble_based_methods.html
a_true = 1.0
b_true = 5.0
number_of_parameters = 2
number_of_observations = 45


class LinearModel:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.size = 2

    @classmethod
    def random(cls):
        a_std = 2.0
        b_std = 2.0
        # Priors with bias
        a_bias = 0.5 * a_std
        b_bias = -0.5 * b_std

        return cls(
            np.random.normal(a_true + a_bias, a_std),
            np.random.normal(b_true + b_bias, b_std),
        )

    def eval(self, x):
        return self.a * x + self.b


def distance(point1, point2):
    """euclidean distance
    :param point1: n-dimensional point (array of floats).
    :param point2: n-dimensional point (array of floats).
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    return np.sqrt(np.sum(np.square(point1 - point2)))


@pytest.mark.parametrize("number_of_realizations", [100, 200])
def test_that_update_for_a_linear_model_works_with_rowscaling(number_of_realizations):
    true_model = LinearModel(a_true, b_true)

    ensemble = [LinearModel.random() for _ in range(number_of_realizations)]

    A = np.array(
        [
            [realization.a for realization in ensemble],
            [realization.b for realization in ensemble],
        ]
    )
    mean_prior = np.mean(A, axis=1)

    # We use time as the x-axis and observations are at
    # t=0,1,2...number_of_observations
    times = np.arange(number_of_observations)

    S = np.array([[realization.eval(t) for realization in ensemble] for t in times])

    # When observations != true model, then ml estimates != true parameters.
    # This gives both a more advanced and realistic test. Standard normal
    # N(0,1) noise is added to obtain this. The randomness ensures we are not
    # gaming the test. But the difference could in principle be any non-zero
    # scalar.
    observations = np.array(
        [true_model.eval(t) + np.random.normal(0.0, 1.0) for t in times]
    )

    # Leading to fixed Maximum likelihood estimate.
    # It will equal true values when observations are sampled without noise.
    # It will also stay the same over beliefs.
    mean_observations = np.mean(observations)
    times_mean = np.mean(times)
    times_square_sum = sum(np.square(times))
    a_maximum_likelihood = sum(
        t * (observations[t] - mean_observations) for t in times
    ) / (times_square_sum - times_mean * sum(times))
    b_maximum_likelihood = mean_observations - a_maximum_likelihood * times_mean
    maximum_likelihood = np.array([a_maximum_likelihood, b_maximum_likelihood])

    previous_mean_posterior = mean_prior

    # numerical precision tolerance
    epsilon = 1e-2

    # We iterate with an increased belief in the observations
    for error in [10000.0, 100.0, 10.0, 1.0, 0.1]:
        # An important point here is that we do not iteratively
        # update A, but instead, observations stay the same and
        # we increase our belief in the observations
        # As A is update inplace, we have to reset it.
        A = np.asfortranarray(
            [
                [realization.a for realization in ensemble],
                [realization.b for realization in ensemble],
            ]
        )
        row_scaling = RowScaling()
        row_scaling[0] = 1.0
        row_scaling[1] = 0.7
        ((A_posterior, _),) = ensemble_smoother_update_step_row_scaling(
            Y=S,
            X_with_row_scaling=[(A, row_scaling)],
            covariance=np.full(observations.shape, error),
            observations=observations,
            seed=42,
        )

        mean_posterior = np.mean(A_posterior, axis=1)

        # All posterior estimates lie between prior and maximum likelihood estimate
        assert np.all(
            distance(mean_posterior, maximum_likelihood)
            - distance(mean_prior, maximum_likelihood)
            < epsilon
        )
        assert np.all(
            distance(mean_prior, mean_posterior)
            - distance(mean_prior, maximum_likelihood)
            < epsilon
        )

        # Posterior parameter estimates improve with increased trust in observations
        assert np.all(
            distance(mean_posterior, maximum_likelihood)
            - distance(previous_mean_posterior, maximum_likelihood)
            < epsilon
        )

        previous_mean_posterior = mean_posterior
