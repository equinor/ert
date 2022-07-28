from unittest.mock import patch

import numpy as np
import pytest
import scipy

import ert
from ert import ert3


def test_gaussian_scalar():
    gauss = ert3.stats.Gaussian(0, 1)

    assert gauss.mean == 0
    assert gauss.std == 1
    assert gauss.size == 1
    sample = gauss.sample()
    assert isinstance(sample.data, float)
    assert sample.record_type == ert.data.RecordType.SCALAR_FLOAT
    assert sample.index == ()


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
    err_msg_both = "Cannot create distribution with both size and index"
    with pytest.raises(ValueError, match=err_msg_both):
        ert3.stats.Gaussian(0, 1, size=10, index=list(range(10)))


def test_uniform_scalar():
    uniform = ert3.stats.Uniform(0, 1)

    assert uniform.lower_bound == 0
    assert uniform.upper_bound == 1
    assert uniform.size == 1
    sample = uniform.sample()
    assert 0 <= sample.data <= 1
    assert sample.record_type == ert.data.RecordType.SCALAR_FLOAT
    assert sample.index == ()


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
    err_msg_both = "Cannot create distribution with both size and index"
    with pytest.raises(ValueError, match=err_msg_both):
        ert3.stats.Uniform(0, 1, size=10, index=list(range(10)))


@pytest.mark.parametrize(
    ("size", "values"),
    (
        (1000, [42]),
        (1000, [42, 42]),
        (1000, [0, 1, 2, 3, 4]),
        (1000, [0, 0.1]),
    ),
)
def test_discrete_validvalues(size, values):
    discrete = ert3.stats.Discrete(values, size=size)

    assert discrete.values == values

    for _ in range(100):
        sample = discrete.sample()
        assert sample.index == tuple(range(size)), "Indices should be identical"
        assert np.isin(sample.data, values).all()


@pytest.mark.parametrize(
    ("index", "values"),
    (
        (("x", "y", "z"), [42]),
        (("z", "y", "x"), [42]),
        ((3, 2, 1), [42]),
        (("x", "y", "z", "blah"), [0, 1, 2, 3, 4]),
    ),
)
def test_discrete_validvalues_index(index, values):
    discrete = ert3.stats.Discrete(values, index=index)

    assert discrete.values == values

    for _ in range(100):
        sample = discrete.sample()
        assert sample.index == index, "Indices should be identical"
        assert np.isin([sample.data[key] for key in index], values).all()


@pytest.mark.parametrize(
    ("mean", "std", "q", "size", "index"),
    (
        (0, 1, 0.005, None, None),
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
    gauss = ert3.stats.Gaussian(mean, std, size=size, index=index)

    expected_value = scipy.stats.norm.ppf(q, loc=mean, scale=std)
    ppf_result = gauss.ppf(q)
    if size is None and index is None:
        assert ppf_result.data == pytest.approx(expected_value)
    else:
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
    dist = ert3.stats.Uniform(lower, upper, size=size, index=index)

    expected_value = scipy.stats.uniform.ppf(q, loc=lower, scale=upper - lower)
    ppf_result = dist.ppf(q)
    if size is None and index is None:
        assert ppf_result.data == pytest.approx(expected_value)
    else:
        assert len(dist.index) == len(ppf_result.data)
        assert sorted(dist.index) == sorted(ppf_result.index)
        for idx in dist.index:
            assert ppf_result.data[idx] == pytest.approx(expected_value)


@pytest.mark.parametrize(
    ("values", "q", "expected_val"),
    (
        ([0], 0.0, np.nan),
        ([0], 0.1, 0),
        ([0], 0.2, 0),
        ([0], 0.5, 0),
        ([0], 0.9, 0),
        ([0], 1.0, 0),
        ([1], 0.9, 1),
        ([0, 0], 0.1, 0),
        ([0, 0], 0.9, 0),
        ([42], 0.999, 42),
        ([0], 1.1, np.nan),
        ([0], 0.1, 0),
        ([42, 420, 4200], 0.0, np.nan),
        ([42, 420, 4200], 0.2, 42),
        ([42, 420, 4200], 0.3333333, 42),
        ([42, 420, 4200], 0.3333334, 420),
        ([42, 420, 4200], 0.5, 420),
        ([42, 420, 4200], 0.6666666, 420),
        ([42, 420, 4200], 0.6666667, 4200),
        ([42, 42, 4200], 0.6666666, 42),
        ([42, 42, 4200], 0.6666667, 4200),
        ([42, 4200, 42], 0.6666666, 42),
        ([42, 4200, 42], 0.6666667, 4200),
        ([42, 420, 4200], 0.9, 4200),
        ([42, 420, 4200], 0.9, 4200),
        ([4200, 420, 42], 0.9, 4200),
        ([42, 420, 4200], 1.1, np.nan),
    ),
)
def test_discrete_ppf(values, q, expected_val):
    # Test distributions with a size argument, yielding implicit indices:
    for size in (1, 5, 10):
        dist = ert3.stats.Discrete(values, size=size)
        assert len(dist.index) == size

        ppf_result = dist.ppf(q)
        assert ppf_result.index == dist.index
        assert np.array_equal(ppf_result.data, [expected_val] * size, equal_nan=True)

    # Test distribution with explicit index values:
    for index in (
        ("a", "b", "c"),
        ("b", "a", "c"),
        ("single_key",),
        tuple(index_len * "x" for index_len in range(1, 10)),
    ):
        dist = ert3.stats.Discrete(values, index=index)

        expected = [expected_val] * len(index)
        assert len(dist.index) == len(expected)

        ppf_result = dist.ppf(q)

        # Sorting is not guaranteed when the values are indexed:
        assert sorted(ppf_result.index) == sorted(dist.index)

        for idx, e in zip(dist.index, expected):
            if np.isnan(e):
                assert np.isnan(ppf_result.data[idx])
            else:
                assert ppf_result.data[idx] == e


@pytest.mark.parametrize(
    ("size", "lower_bound", "upper_bound"),
    (
        (10000, 0.001, 1),
        (20000, 10, 20),
    ),
)
def test_loguniform_samples_unique(size, lower_bound, upper_bound):
    loguniform = ert3.stats.Uniform(lower_bound, upper_bound, size=size)

    assert loguniform.lower_bound == lower_bound
    assert loguniform.upper_bound == upper_bound

    # Essentially tests that samples drawn from Uniform are unique
    prev_samples = set()
    for _ in range(100):
        sample = loguniform.sample()

        assert loguniform.index == tuple(range(size)), "Indices should be identical"
        assert tuple(sample.data) not in prev_samples, "Samples should be different"

        prev_samples.add(tuple(sample.data))


@pytest.mark.parametrize(
    ("size", "lower_bound", "upper_bound"),
    (
        (10000, 0.001, 1),
        (20000, 10, 20),
    ),
)
def test_loguniform_samples_comesfromscipy(size, lower_bound, upper_bound):
    with patch("scipy.stats.loguniform.rvs") as scipy:
        retval = np.array(range(100))
        scipy.return_value = retval  # never mind the content

        loguniform = ert3.stats.LogUniform(lower_bound, upper_bound, size=size)
        data = loguniform.sample().data

        scipy.assert_called_once_with(a=lower_bound, b=upper_bound, size=size)
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
def test_loguniform_index(index, lower_bound, upper_bound):
    with patch("scipy.stats.loguniform.rvs") as scipy:
        loguniform = ert3.stats.LogUniform(lower_bound, upper_bound, index=index)

        assert loguniform.lower_bound == lower_bound
        assert loguniform.upper_bound == upper_bound

        samples = {idx: [] for idx in index}
        for i in range(100):

            scipy.return_value = [i] * len(index)
            sample = loguniform.sample()
            scipy.assert_called_once_with(a=lower_bound, b=upper_bound, size=len(index))
            scipy.reset_mock()

            assert sorted(loguniform.index) == sorted(
                index
            ), "Indices should be the same"
            assert sorted(sample.index) == sorted(index), "Indices should be the same"
            for key in index:
                samples[key].append(sample.data[key])

    for key in index:
        s = np.array(samples[key])
        assert np.alltrue(s == range(100)), f"Wrong samples for key {key}"


def test_loguniform_distribution_invalid():
    err_msg_both = "Cannot create distribution with both size and index"
    with pytest.raises(ValueError, match=err_msg_both):
        ert3.stats.LogUniform(0.001, 1, size=10, index=list(range(10)))


@pytest.mark.parametrize(
    ("lower", "upper", "q", "size", "index"),
    (
        (0.001, 2, 0.5, 1, None),
        (2, 5, 0.1, 5, None),
        (1, 2, 0.7, 10, None),
        (0.001, 1, 0.005, None, ("a", "b", "c")),
        (0.001, 1, 0.995, None, tuple(index_len * "x" for index_len in range(1, 10))),
        (2, 5, 0.1, None, ("x", "y")),
        (1, 2, 0.7, None, ("single_key",)),
    ),
)
def test_loguniform_ppf(lower, upper, q, size, index):
    if size is not None:
        dist = ert3.stats.LogUniform(lower, upper, size=size)
    else:
        dist = ert3.stats.LogUniform(lower, upper, index=index)

    expected_value = scipy.stats.loguniform.ppf(q, a=lower, b=upper)
    ppf_result = dist.ppf(q)
    assert len(dist.index) == len(ppf_result.data)
    assert sorted(dist.index) == sorted(ppf_result.index)
    for idx in dist.index:
        assert ppf_result.data[idx] == pytest.approx(expected_value)
