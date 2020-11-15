import ert3

import flaky
import numpy as np
import pytest


def approx(x, eps=0.2):
    return pytest.approx(x, abs=eps, rel=eps)


@flaky.flaky(max_runs=3, min_passes=2)
@pytest.mark.parametrize(
    ("size", "mean", "std"),
    (
        (30000, 0, 1),
        (30000, 10, 10),
    ),
)
def test_gaussian_distribution(size, mean, std):
    gauss = ert3.stats.Gaussian(mean, std, size=size)

    prev_samples = set()
    for _ in range(10):
        sample = gauss.sample()

        assert len(sample) == size

        assert tuple(sample) not in prev_samples
        prev_samples.add(tuple(sample))

        assert sample.mean() == approx(mean)
        assert sample.std() == approx(std)


@flaky.flaky(max_runs=3, min_passes=2)
@pytest.mark.parametrize(
    ("index", "mean", "std"),
    (
        (("a", "b", "c"), 0, 1),
        (tuple("a" * i for i in range(1, 10)), 2, 5),
    ),
)
def test_gaussian_distribution_index(index, mean, std):
    gauss = ert3.stats.Gaussian(mean, std, index=index)

    samples = {idx: [] for idx in index}
    for i in range(2000):
        sample = gauss.sample()
        assert sorted(sample.keys()) == sorted(index)

        for key in index:
            samples[key].append(sample[key])

    for key in index:
        s = np.array(samples[key])
        assert s.mean() == approx(mean)
        assert s.std() == approx(std)


def test_gaussian_distribution_invalid():
    err_msg_neither = "Cannot create gaussian distribution with neither size nor index"
    with pytest.raises(ValueError, match=err_msg_neither):
        ert3.stats.Gaussian(0, 1)

    err_msg_both = "Cannot create gaussian distribution with both size and index"
    with pytest.raises(ValueError, match=err_msg_both):
        ert3.stats.Gaussian(0, 1, size=10, index=list(range(10)))


@flaky.flaky(max_runs=3, min_passes=2)
@pytest.mark.parametrize(
    ("size", "lower_bound", "upper_bound"),
    (
        (10000, 0, 1),
        (20000, 10, 20),
    ),
)
def test_uniform_distribution(size, lower_bound, upper_bound):
    uniform = ert3.stats.Uniform(lower_bound, upper_bound, size=size)

    prev_samples = set()
    for _ in range(10):
        sample = uniform.sample()

        assert len(sample) == size

        assert tuple(sample) not in prev_samples
        prev_samples.add(tuple(sample))

        assert sample.min() == approx(lower_bound)
        assert sample.mean() == approx((lower_bound + upper_bound) / 2)
        assert sample.max() == approx(upper_bound)


@flaky.flaky(max_runs=3, min_passes=2)
@pytest.mark.parametrize(
    ("index", "lower_bound", "upper_bound"),
    (
        (("a", "b", "c"), 0, 1),
        (tuple("a" * i for i in range(1, 10)), 2, 5),
    ),
)
def test_uniform_distribution_index(index, lower_bound, upper_bound):
    uniform = ert3.stats.Uniform(lower_bound, upper_bound, index=index)

    samples = {idx: [] for idx in index}
    for i in range(1000):
        sample = uniform.sample()
        assert sorted(sample.keys()) == sorted(index)

        for key in index:
            samples[key].append(sample[key])

    for key in index:
        s = np.array(samples[key])
        assert s.min() == approx(lower_bound)
        assert s.mean() == approx((lower_bound + upper_bound) / 2)
        assert s.max() == approx(upper_bound)


def test_uniform_distribution_invalid():
    err_msg_neither = "Cannot create uniform distribution with neither size nor index"
    with pytest.raises(ValueError, match=err_msg_neither):
        ert3.stats.Uniform(0, 1)

    err_msg_both = "Cannot create uniform distribution with both size and index"
    with pytest.raises(ValueError, match=err_msg_both):
        ert3.stats.Uniform(0, 1, size=10, index=list(range(10)))
