import asyncio
from typing import Any, Dict, List, MutableMapping

import numpy as np
import pytest

import ert
import ert3

# Expected values for S1 and ST for ishigami function
# sample_size = 15000, harmonics = 4, bounds = [-pi, pi]
# uniform distribution
from ert_shared.async_utils import get_event_loop

ISHIGAMI_S1 = [0.3076, 0.4424, 6.351e-27]
ISHIGAMI_ST = [0.5507, 0.4695, 0.2392]


def ishigami_single(x, y, z, A=7, B=0.1) -> float:
    return np.sin(x) + A * np.power(np.sin(y), 2) + B * np.power(z, 4) * np.sin(x)


def ishigami_multiple(x, y, z, A=7, B=0.1) -> List[float]:
    return [
        np.sin(x) + A * np.power(np.sin(y), 2) + B * np.power(z, 4) * np.sin(x)
    ] * 10


def polynomial(x, y, a, b, c):
    return [a * x**2 + b * y + c]


def assert_samples(
    samples: List[Dict[str, Any]],
    sample_size: int,
    param_size: int,
    parameters: MutableMapping[str, ert3.stats.Distribution],
):
    """Verifies dimensions of computed samples."""
    assert sample_size * param_size == len(samples)
    for records in samples:
        assert records.keys() == parameters.keys()
        for key in records.keys():
            if parameters[key].is_scalar:
                assert isinstance(records[key].data, np.float64)
            else:
                assert (
                    list(records[key].data.keys())
                    == list(records[key].index)
                    == list(parameters[key].index)
                )


def assert_analysis(analysis, analysis_size, param_size, param_index):
    assert analysis_size == len(analysis)
    for i in range(analysis_size):
        assert 5 == len(analysis[i])
        assert (
            param_size
            == len(analysis[i]["S1"])
            == len(analysis[i]["ST"])
            == len(analysis[i]["S1_conf"])
            == len(analysis[i]["ST_conf"])
        )
        assert analysis[i]["names"] == list(map(str, param_index))


@pytest.mark.parametrize(
    ("distribution"),
    (
        (ert3.stats.Gaussian(mean=-np.pi, std=np.pi)),
        (ert3.stats.Uniform(lower_bound=-np.pi, upper_bound=np.pi)),
        (ert3.stats.Gaussian(mean=-np.pi, std=np.pi, index=tuple(["x", "y", "z"]))),
        (
            ert3.stats.Uniform(
                lower_bound=-np.pi, upper_bound=np.pi, index=tuple(["x", "y", "z"])
            )
        ),
    ),
)
def test_single_evaluation(distribution):
    sample_size = 15000
    harmonics = 4
    analysis_size = 1
    parameters = {"xs": distribution}

    samples = ert3.algorithms.fast_sample(parameters, harmonics, sample_size)

    # Verify dimensions of returned samples:
    assert_samples(samples, sample_size, distribution.size, parameters)

    if distribution.size == 1:
        # A single scalar record as input does not make much sense in FAST, skip
        # running the model and the associated output analysis
        return

    model_output = {}
    futures = []

    for iens, sample in enumerate(samples):
        x = sample["xs"].data["x"]
        y = sample["xs"].data["y"]
        z = sample["xs"].data["z"]
        t = ert.data.InMemoryRecordTransmitter("output")
        futures.append(
            t.transmit_record(ert.data.NumericalRecord(data=[ishigami_single(x, y, z)]))
        )
        model_output[iens] = {"output": t}
    get_event_loop().run_until_complete(asyncio.gather(*futures))

    analysis = ert3.algorithms.fast_analyze(parameters, model_output, harmonics)
    assert_analysis(analysis, analysis_size, distribution.size, parameters["xs"].index)

    if distribution.type == "uniform":
        S1 = analysis[0]["S1"]
        ST = analysis[0]["ST"]
        np.testing.assert_allclose(S1, ISHIGAMI_S1, 1e-4, 1e-4)
        np.testing.assert_allclose(ST, ISHIGAMI_ST, 1e-4, 1e-4)


@pytest.mark.parametrize(
    ("distribution"),
    (
        (ert3.stats.Gaussian(mean=-np.pi, std=np.pi)),
        (ert3.stats.Uniform(lower_bound=-np.pi, upper_bound=np.pi)),
    ),
)
def test_scalar_distributions(distribution):
    sample_size = 15000
    harmonics = 4
    analysis_size = 1
    parameters = {"x": distribution, "y": distribution, "z": distribution}
    samples = ert3.algorithms.fast_sample(parameters, harmonics, sample_size)
    assert_samples(samples, sample_size, len(parameters), parameters)

    model_output = {}
    futures = []

    for iens, sample in enumerate(samples):
        x = sample["x"].data
        y = sample["y"].data
        z = sample["z"].data
        t = ert.data.InMemoryRecordTransmitter("output")
        futures.append(
            t.transmit_record(ert.data.NumericalRecord(data=ishigami_single(x, y, z)))
        )
        model_output[iens] = {"output": t}
    get_event_loop().run_until_complete(asyncio.gather(*futures))

    analysis = ert3.algorithms.fast_analyze(parameters, model_output, harmonics)
    assert_analysis(analysis, analysis_size, len(parameters), parameters.keys())

    if distribution.type == "uniform":
        S1 = analysis[0]["S1"]
        ST = analysis[0]["ST"]
        np.testing.assert_allclose(S1, ISHIGAMI_S1, 1e-4, 1e-4)
        np.testing.assert_allclose(ST, ISHIGAMI_ST, 1e-4, 1e-4)


def test_parameter_array():
    sample_size = 15000
    harmonics = 4
    param_size = 3
    analysis_size = 1
    parameters = {
        "xs": ert3.stats.Uniform(lower_bound=-np.pi, upper_bound=np.pi, size=3)
    }

    samples = ert3.algorithms.fast_sample(parameters, harmonics, sample_size)
    assert_samples(samples, sample_size, param_size, parameters)

    model_output = {}
    futures = []
    for iens, sample in enumerate(samples):
        x = sample["xs"].data[0]
        y = sample["xs"].data[1]
        z = sample["xs"].data[2]
        t = ert.data.InMemoryRecordTransmitter("output")
        futures.append(
            t.transmit_record(ert.data.NumericalRecord(data=[ishigami_single(x, y, z)]))
        )
        model_output[iens] = {"output": t}
    get_event_loop().run_until_complete(asyncio.gather(*futures))

    analysis = ert3.algorithms.fast_analyze(parameters, model_output, harmonics)
    assert_analysis(analysis, analysis_size, param_size, parameters["xs"].index)

    S1 = analysis[0]["S1"]
    ST = analysis[0]["ST"]
    np.testing.assert_allclose(S1, ISHIGAMI_S1, 1e-4, 1e-4)
    np.testing.assert_allclose(ST, ISHIGAMI_ST, 1e-4, 1e-4)


def test_multiple_evaluations():
    sample_size = 15000
    harmonics = 4
    param_size = 3
    analysis_size = 10
    parameters = {
        "xs": ert3.stats.Uniform(
            lower_bound=-np.pi, upper_bound=np.pi, index=tuple(["x", "y", "z"])
        )
    }
    samples = ert3.algorithms.fast_sample(parameters, harmonics, sample_size)
    assert_samples(samples, sample_size, param_size, parameters)

    model_output = {}
    futures = []
    for iens, sample in enumerate(samples):
        x = sample["xs"].data["x"]
        y = sample["xs"].data["y"]
        z = sample["xs"].data["z"]
        t = ert.data.InMemoryRecordTransmitter("output")
        futures.append(
            t.transmit_record(ert.data.NumericalRecord(data=ishigami_multiple(x, y, z)))
        )
        model_output[iens] = {"output": t}
    get_event_loop().run_until_complete(asyncio.gather(*futures))

    analysis = ert3.algorithms.fast_analyze(parameters, model_output, harmonics)
    assert_analysis(analysis, analysis_size, param_size, parameters["xs"].index)

    for i in range(analysis_size):
        S1 = analysis[i]["S1"]
        ST = analysis[i]["ST"]
        np.testing.assert_allclose(S1, ISHIGAMI_S1, 1e-4, 1e-4)
        np.testing.assert_allclose(ST, ISHIGAMI_ST, 1e-4, 1e-4)


def test_mixed_record_type():
    sample_size = 15000
    harmonics = 4
    param_size = 3
    analysis_size = 10
    parameters = {
        "x": ert3.stats.Uniform(lower_bound=-np.pi, upper_bound=np.pi),
        "yz": ert3.stats.Uniform(
            lower_bound=-np.pi, upper_bound=np.pi, index=tuple(["y", "z"])
        ),
    }
    samples = ert3.algorithms.fast_sample(parameters, harmonics, sample_size)
    assert_samples(samples, sample_size, param_size, parameters)

    model_output = {}
    futures = []
    for iens, sample in enumerate(samples):
        x = sample["x"].data
        y = sample["yz"].data["y"]
        z = sample["yz"].data["z"]
        t = ert.data.InMemoryRecordTransmitter("output")
        futures.append(
            t.transmit_record(ert.data.NumericalRecord(data=ishigami_multiple(x, y, z)))
        )
        model_output[iens] = {"output": t}
    get_event_loop().run_until_complete(asyncio.gather(*futures))

    analysis = ert3.algorithms.fast_analyze(parameters, model_output, harmonics)
    assert_analysis(analysis, analysis_size, param_size, ["x", "y", "z"])

    for i in range(analysis_size):
        S1 = analysis[i]["S1"]
        ST = analysis[i]["ST"]
        np.testing.assert_allclose(S1, ISHIGAMI_S1, 1e-4, 1e-4)
        np.testing.assert_allclose(ST, ISHIGAMI_ST, 1e-4, 1e-4)


def test_analyse_multiple_groups():
    sample_size = 15000
    harmonics = 4
    param_size = 5
    analysis_size = 1
    parameters = {
        "xs": ert3.stats.Uniform(lower_bound=0, upper_bound=1, index=tuple(["x", "y"])),
        "coeffs": ert3.stats.Gaussian(mean=0, std=1, index=tuple(["a", "b", "c"])),
    }
    samples = ert3.algorithms.fast_sample(parameters, harmonics, sample_size)
    assert_samples(samples, sample_size, param_size, parameters)

    model_output = {}
    futures = []
    for iens, sample in enumerate(samples):
        x = sample["xs"].data["x"]
        y = sample["xs"].data["y"]
        a = sample["coeffs"].data["a"]
        b = sample["coeffs"].data["b"]
        c = sample["coeffs"].data["c"]
        evaluation = polynomial(x, y, a, b, c)
        t = ert.data.InMemoryRecordTransmitter("output")
        futures.append(t.transmit_record(ert.data.NumericalRecord(data=evaluation)))
        model_output[iens] = {"output": t}

    get_event_loop().run_until_complete(asyncio.gather(*futures))

    analysis = ert3.algorithms.fast_analyze(parameters, model_output, harmonics)
    assert_analysis(
        analysis,
        analysis_size,
        param_size,
        list(parameters["xs"].index + parameters["coeffs"].index),
    )


@pytest.mark.parametrize(("sample_size"), ((10000), (200)))
def test_sample_size(sample_size):
    harmonics = 4
    param_size = 3
    analysis_size = 1
    parameters = {
        "xs": ert3.stats.Uniform(
            lower_bound=-np.pi, upper_bound=np.pi, index=tuple(["x", "y", "z"])
        )
    }
    samples = ert3.algorithms.fast_sample(parameters, harmonics, sample_size)
    assert_samples(samples, sample_size, param_size, parameters)

    model_output = {}
    futures = []
    for iens, sample in enumerate(samples):
        x = sample["xs"].data["x"]
        y = sample["xs"].data["y"]
        z = sample["xs"].data["z"]
        t = ert.data.InMemoryRecordTransmitter("output")
        futures.append(
            t.transmit_record(ert.data.NumericalRecord(data=[ishigami_single(x, y, z)]))
        )
        model_output[iens] = {"output": t}
    get_event_loop().run_until_complete(asyncio.gather(*futures))

    analysis = ert3.algorithms.fast_analyze(parameters, model_output, harmonics)
    assert_analysis(analysis, analysis_size, param_size, parameters["xs"].index)
