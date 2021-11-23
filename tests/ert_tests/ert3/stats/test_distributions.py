from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy

import ert3


@pytest.mark.parametrize(
    ("size", "mean", "std"),
    (
        (30000, 0, 1),
        (30000, 10, 10),
    ),
)
def test_gaussian_samples_unique(size, mean, std):
    gauss = ert3.stats.Gaussian(mean, std, size=size)

    assert gauss.mean == mean
    assert gauss.std == std

    # Essentially tests that samples drawn from Gaussian are unique
    prev_samples = set()
    for _ in range(100):
        sample = gauss.sample()

        assert gauss.index == tuple(range(size)), "Indices should be identical"
        assert tuple(sample.data) not in prev_samples, "Samples should be unique"

        prev_samples.add(tuple(sample.data))


@pytest.mark.parametrize(
    ("size", "mean", "std"),
    (
        (30000, 0, 1),
        (30000, 10, 10),
    ),
)
def test_gaussian_samples_comesfromscipy(size, mean, std):
    with patch("scipy.stats.norm.rvs") as scipy:
        retval = np.array(range(100))
        scipy.return_value = retval  # never mind the content

        gauss = ert3.stats.Gaussian(mean, std, size=size)
        data = gauss.sample().data

        scipy.assert_called_once_with(loc=mean, scale=std, size=size)
        assert np.array_equal(
            retval, data
        ), "Samples should be identical to rvs from Scipy"


@pytest.mark.parametrize(
    ("index", "mean", "std"),
    (
        (("a", "b", "c"), 0, 1),
        (tuple("a" * i for i in range(1, 10)), 2, 5),
    ),
)
def test_gaussian_index(index, mean, std):
    with patch("scipy.stats.norm.rvs") as scipy:
        gauss = ert3.stats.Gaussian(mean, std, index=index)

        assert gauss.mean == mean
        assert gauss.std == std

        samples = {idx: [] for idx in index}
        for i in range(100):

            # The return-value is an array of "i"
            scipy.return_value = [i] * len(index)
            sample = gauss.sample()
            scipy.assert_called_once_with(loc=mean, scale=std, size=len(index))
            scipy.reset_mock()

            assert sorted(gauss.index) == sorted(index), "Indices should be the same"
            assert sorted(sample.index) == sorted(index), "Indices should be the same"

            for key in index:
                samples[key].append(sample.data[key])

    for key in index:
        s = np.array(samples[key])
        assert np.alltrue(s == range(100)), f"Wrong samples for key {key}"


def test_gaussian_distribution_invalid():
    err_msg_neither = "Cannot create distribution with neither size nor index"
    with pytest.raises(ValueError, match=err_msg_neither):
        ert3.stats.Gaussian(0, 1)

    err_msg_both = "Cannot create distribution with both size and index"
    with pytest.raises(ValueError, match=err_msg_both):
        ert3.stats.Gaussian(0, 1, size=10, index=list(range(10)))


@pytest.mark.parametrize(
    ("size", "lower_bound", "upper_bound"),
    (
        (10000, 0, 1),
        (20000, 10, 20),
    ),
)
def test_uniform_samples_unique(size, lower_bound, upper_bound):
    uniform = ert3.stats.Uniform(lower_bound, upper_bound, size=size)

    assert uniform.lower_bound == lower_bound
    assert uniform.upper_bound == upper_bound

    # Essentially tests that samples drawn from Uniform are unique
    prev_samples = set()
    for _ in range(100):
        sample = uniform.sample()

        assert uniform.index == tuple(range(size)), "Indices should be identical"
        assert tuple(sample.data) not in prev_samples, "Samples should be different"

        prev_samples.add(tuple(sample.data))


@pytest.mark.parametrize(
    ("size", "lower_bound", "upper_bound"),
    (
        (10000, 0, 1),
        (20000, 10, 20),
    ),
)
def test_uniform_samples_comesfromscipy(size, lower_bound, upper_bound):
    with patch("scipy.stats.uniform.rvs") as scipy:
        retval = np.array(range(100))
        scipy.return_value = retval  # never mind the content

        uniform = ert3.stats.Uniform(lower_bound, upper_bound, size=size)
        data = uniform.sample().data

        scipy.assert_called_once_with(
            loc=lower_bound, scale=(upper_bound - lower_bound), size=size
        )
        assert np.array_equal(
            retval, data
        ), "Samples should be identical to rvs from Scipy"


@pytest.mark.parametrize(
    ("index", "lower_bound", "upper_bound"),
    (
        (("a", "b", "c"), 0, 1),
        (tuple("a" * i for i in range(1, 10)), 2, 5),
    ),
)
def test_uniform_index(index, lower_bound, upper_bound):
    with patch("scipy.stats.uniform.rvs") as scipy:
        uniform = ert3.stats.Uniform(lower_bound, upper_bound, index=index)

        assert uniform.lower_bound == lower_bound
        assert uniform.upper_bound == upper_bound

        samples = {idx: [] for idx in index}
        for i in range(100):

            scipy.return_value = [i] * len(index)
            sample = uniform.sample()
            scipy.assert_called_once_with(
                loc=lower_bound, scale=(upper_bound - lower_bound), size=len(index)
            )
            scipy.reset_mock()

            assert sorted(uniform.index) == sorted(index), "Indices should be the same"
            assert sorted(sample.index) == sorted(index), "Indices should be the same"
            for key in index:
                samples[key].append(sample.data[key])

    for key in index:
        s = np.array(samples[key])
        assert np.alltrue(s == range(100)), f"Wrong samples for key {key}"


def test_uniform_distribution_invalid():
    err_msg_neither = "Cannot create distribution with neither size nor index"
    with pytest.raises(ValueError, match=err_msg_neither):
        ert3.stats.Uniform(0, 1)

    err_msg_both = "Cannot create distribution with both size and index"
    with pytest.raises(ValueError, match=err_msg_both):
        ert3.stats.Uniform(0, 1, size=10, index=list(range(10)))


@pytest.mark.parametrize(
    ("mean", "std", "q", "size", "index"),
    (
        (0, 1, 0.005, 3, None),
        (0, 1, 0.995, 1, None),
        (2, 5, 0.1, 5, None),
        (1, 2, 0.7, 10, None),
        (0, 1, 0.005, None, ("a", "b", "c")),
        (0, 1, 0.995, None, tuple(index_len * "x" for index_len in range(1, 10))),
        (2, 5, 0.1, None, ("x", "y")),
        (1, 2, 0.7, None, ("single_key",)),
    ),
)
def test_gaussian_ppf(mean, std, q, size, index):
    if size is not None:
        gauss = ert3.stats.Gaussian(mean, std, size=size)
    else:
        gauss = ert3.stats.Gaussian(mean, std, index=index)

    expected_value = scipy.stats.norm.ppf(q, loc=mean, scale=std)
    ppf_result = gauss.ppf(q)
    assert len(gauss.index) == len(ppf_result.data)
    assert sorted(gauss.index) == sorted(ppf_result.index)
    for idx in gauss.index:
        assert ppf_result.data[idx] == pytest.approx(expected_value)


@pytest.mark.parametrize(
    ("lower", "upper", "q", "size", "index"),
    (
        (0, 2, 0.5, 1, None),
        (2, 5, 0.1, 5, None),
        (1, 2, 0.7, 10, None),
        (0, 1, 0.005, None, ("a", "b", "c")),
        (0, 1, 0.995, None, tuple(index_len * "x" for index_len in range(1, 10))),
        (2, 5, 0.1, None, ("x", "y")),
        (1, 2, 0.7, None, ("single_key",)),
    ),
)
def test_uniform_ppf(lower, upper, q, size, index):
    if size is not None:
        dist = ert3.stats.Uniform(lower, upper, size=size)
    else:
        dist = ert3.stats.Uniform(lower, upper, index=index)

    expected_value = scipy.stats.uniform.ppf(q, loc=lower, scale=upper - lower)
    ppf_result = dist.ppf(q)
    assert len(dist.index) == len(ppf_result.data)
    assert sorted(dist.index) == sorted(ppf_result.index)
    for idx in dist.index:
        assert ppf_result.data[idx] == pytest.approx(expected_value)
