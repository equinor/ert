import asyncio
from typing import Set, MutableMapping, List, Dict, Optional, Any

import numpy as np
from SALib.sample import fast_sampler
from SALib.analyze import fast

from ert3.stats import Distribution, Gaussian, Uniform
from ert.data import RecordTransmitter, Record, NumericalRecord


def _build_base_records(
    groups: Set[str], parameters: MutableMapping[str, Distribution]
) -> Dict[str, Record]:
    return {group_name: parameters[group_name].ppf(0.5) for group_name in groups}


def _build_salib_problem(
    parameters: MutableMapping[str, Distribution]
) -> Dict[str, Any]:
    # SALib problem structure is described in
    # https://salib.readthedocs.io/en/latest/basics.html

    problem = {}
    dists = parameters.values()
    num_vars = sum(dist.size for dist in dists)
    names = [str(name) for idx in (list(dist.index) for dist in dists) for name in idx]

    bounds = []
    types = []
    for dist in dists:
        assert isinstance(dist, (Uniform, Gaussian))  # To make mypy checker happy
        if dist.type == "uniform":
            assert isinstance(dist, Uniform)  # To make mypy checker happy
            bounds.extend([[dist.lower_bound, dist.upper_bound]] * dist.size)
            types.extend(["unif"] * dist.size)
        elif dist.type == "gaussian":
            assert isinstance(dist, Gaussian)  # To make mypy checker happy
            bounds.extend([[dist.mean, dist.std]] * dist.size)
            types.extend(["norm"] * dist.size)
        else:
            raise ValueError(f"Unsupported distribution type {dist.type}")

    problem["num_vars"] = num_vars
    problem["names"] = names  # type: ignore
    problem["bounds"] = bounds  # type: ignore
    problem["dists"] = types  # type: ignore

    return problem


def one_at_the_time(
    parameters: MutableMapping[str, Distribution], tail: Optional[float] = 0.99
) -> List[Dict[str, Record]]:
    if len(parameters) == 0:
        raise ValueError("Cannot study the sensitivity of no variables")

    # The lower extremal value of the analysis. Each variable will after turn
    # be explored as ppf(q) and ppf(1-q) as the two extremal values.
    if tail is not None:
        q = (1 - tail) / 2
    else:
        q = (1 - 0.99) / 2

    evaluations = []
    for group_name, dist in parameters.items():
        lower = dist.ppf(q)
        upper = dist.ppf(1 - q)
        const_records = set(parameters.keys())
        const_records.remove(group_name)

        for idx in dist.index:
            for lim_val in (lower, upper):
                records = _build_base_records(const_records, parameters)

                rec_values = dist.ppf(0.5)
                rec_values.data[idx] = lim_val.data[idx]  # type: ignore
                records[group_name] = rec_values

                evaluations.append(records)

    return evaluations


def fast_sample(
    parameters: MutableMapping[str, Distribution],
    harmonics: Optional[int],
    sample_size: Optional[int],
) -> List[Dict[str, Record]]:
    if len(parameters) == 0:
        raise ValueError("Cannot study the sensitivity of no variables")

    if harmonics is None:
        harmonics = 4
    if sample_size is None:
        sample_size = 1000

    problem = _build_salib_problem(parameters)

    samples = fast_sampler.sample(problem, sample_size, M=harmonics)

    group_records = []
    for dist in parameters.values():
        records = []
        for sample in samples:
            data = dict(zip(dist.index, sample[: dist.size]))
            record = NumericalRecord(data=data, index=dist.index)
            records.append(record)
        samples = np.delete(samples, list(range(dist.size)), axis=1)  # type: ignore
        group_records.append(records)

    evaluations = []
    for i in zip(*group_records):
        evaluation = dict(zip(parameters.keys(), i))
        evaluations.append(evaluation)

    return evaluations


def fast_analyze(
    parameters: MutableMapping[str, Distribution],
    model_output: Dict[int, Dict[str, RecordTransmitter]],
    harmonics: Optional[int],
) -> Dict[int, Dict[str, Any]]:

    # Returns a dictionary with S1, ST, S1_conf, ST_conf and names
    # (described in https://salib.readthedocs.io/en/latest/api.html)
    # for each evaluation performed with a sample, i.e. a polynomial
    # is evaluated with a set of sample coefficients for 10 values of x
    # ie. {0: {'S1': [0.3075(x), 0.4424(y), 4.531e-27(c)], ...,
    # 'names': ['x', 'y', 'z']}, 1: ...}

    if len(parameters) == 0:
        raise ValueError("Cannot study the sensitivity of no variables")
    records = []
    for transmitter_map in model_output.values():
        if len(transmitter_map) > 1:
            raise ValueError("Cannot analyze sensitivity with multiple outputs")
        if len(transmitter_map) < 1:
            raise ValueError("Cannot analyze sensitivity with no output")
        for transmitter in transmitter_map.values():
            records.append(
                asyncio.get_event_loop().run_until_complete(transmitter.load())
            )

    ensemble_size = len(model_output)
    if harmonics is None:
        harmonics = 4

    param_size = sum(dist.size for dist in parameters.values())

    if ensemble_size % param_size == 0:
        sample_size = int(ensemble_size / param_size)
    else:
        raise ValueError(
            "The size of the model output must be "
            "a multiple of the number of parameters"
        )

    record_size = len(records[0].data)
    data = np.zeros([sample_size * param_size, record_size])
    for i, record in enumerate(records):
        for j in range(record_size):
            data[i][j] = record.data[j]  # type: ignore

    problem = _build_salib_problem(parameters)

    analysis = {}
    for j in range(record_size):
        analysis[j] = fast.analyze(problem, data[:, j], M=harmonics)

    return analysis
