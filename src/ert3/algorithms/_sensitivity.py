from typing import Any, Dict, List, MutableMapping, Optional, Set

import numpy as np
from SALib.analyze import fast
from SALib.sample import fast_sampler

from ert3.stats import Distribution, Gaussian, Uniform
from ert.data import NumericalRecord, Record, RecordTransmitter, RecordType
from ert_shared.async_utils import get_event_loop

SALIB_DISTS = {"uniform": "unif", "gaussian": "norm", "loguniform": None}
"""Map from ERT3 distribution names to SALib distribution names. If mapped to
None, the distribution is not supported in SALib"""


def _build_base_records(
    parameter_names: Set[str], parameters: MutableMapping[str, Distribution]
) -> Dict[str, Record]:
    """
    Construct "base-case" records, that is, each requested parameter is evalutated
    at the 50% percentile.
    """
    return {
        parameter_name: parameters[parameter_name].ppf(0.5)
        for parameter_name in parameter_names
    }


def _build_salib_problem(
    parameters: MutableMapping[str, Distribution]
) -> Dict[str, Any]:
    """Build a SALib problem structure according to description in
    https://salib.readthedocs.io/en/latest/basics.html

    The returned structure does not differentiate between size-1 parameters and
    scalar parameters.
    """

    problem = {}
    dists = parameters.values()
    num_vars = sum(dist.size for dist in dists)
    names = []
    for param_name, dist in parameters.items():
        if dist.is_scalar:
            names.append(param_name)
        else:
            names.extend([str(name) for name in dist.index])

    bounds = []
    types = []
    for dist in dists:
        if not isinstance(dist, (Gaussian, Uniform)):
            raise ValueError(
                "Only Gaussian and Uniform distributions are supported by "
                f"sensitivity algorightms in ERT, got {dist}."
            )
        if dist.type == "uniform":
            assert isinstance(dist, Uniform)  # To make mypy checker happy
            bounds.extend([[dist.lower_bound, dist.upper_bound]] * dist.size)
            types.extend([SALIB_DISTS["uniform"]] * dist.size)
        elif dist.type == "gaussian":
            assert isinstance(dist, Gaussian)  # To make mypy checker happy
            bounds.extend([[dist.mean, dist.std]] * dist.size)
            types.extend([SALIB_DISTS["gaussian"]] * dist.size)

    problem["num_vars"] = num_vars
    problem["names"] = names  # type: ignore
    problem["bounds"] = bounds  # type: ignore
    problem["dists"] = types  # type: ignore

    return problem


def one_at_a_time(
    parameters: MutableMapping[str, Distribution], tail: Optional[float] = 0.99
) -> List[Dict[str, Record]]:
    if len(parameters) == 0:
        raise ValueError("Cannot study the sensitivity of no variables")

    # Each variable will in turn # be explored as ppf(q) and ppf(1-q) as the two
    # extremal values.

    if tail is not None:
        q = (1 - tail) / 2
    else:
        q = (1 - 0.99) / 2

    evaluations = []
    for parameter_name, dist in parameters.items():
        lower = dist.ppf(q)
        upper = dist.ppf(1 - q)
        constant_records = set(parameters.keys())
        constant_records.remove(parameter_name)

        if dist.is_scalar:
            for lim_val in (lower, upper):
                records = _build_base_records(constant_records, parameters)
                assert isinstance(lim_val, NumericalRecord)
                rec_value = NumericalRecord(data=lim_val.data)
                records[parameter_name] = rec_value
                evaluations.append(records)
        else:
            for idx in dist.index:
                for lim_val in (lower, upper):
                    # Non-scalar parameters are treated as multiple parameters,
                    # an explicit evaluation will be created for each variable
                    records = _build_base_records(constant_records, parameters)

                    rec_values = dist.ppf(0.5)

                    # Modify only one of the base record values:
                    rec_values.data[idx] = lim_val.data[idx]  # type: ignore
                    records[parameter_name] = rec_values

                    evaluations.append(records)

    return evaluations


def fast_sample(
    parameters: MutableMapping[str, Distribution],
    harmonics: Optional[int],
    sample_size: Optional[int],
) -> List[Dict[str, Record]]:
    """Construct the Numpy matrix with model inputs for the extended Fourier
    Amplitude Sensitivity test. The generated samples are intended to be used by
    fast_analyze() after a model has been run the returned data from this
    function.

    This function essentially wraps SALib.sample.fast_sampler.sample.

    Args:
        parameters: Collection of ert3 distributions
        harmonics: The inference parameter, in SALib called "M"
        sample_size: Number of samples to generate, called "N" in SALib
    """
    if len(parameters) == 0:
        raise ValueError("Cannot study the sensitivity without any variables")

    if harmonics is None:
        harmonics = 4
    if sample_size is None:
        sample_size = 1000

    problem = _build_salib_problem(parameters)

    samples = fast_sampler.sample(problem, sample_size, M=harmonics)
    group_records = []
    for dist in parameters.values():
        # Loop over each parameter, let it be scalar or list-like
        records = []
        for sample in samples:
            if dist.is_scalar:
                data = sample[0]
            else:
                data = dict(zip(dist.index, sample[: dist.size]))
            record = NumericalRecord(data=data, index=dist.index)
            records.append(record)
        samples = np.delete(samples, list(range(dist.size)), axis=1)
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
    """
    Perform a sensitivity analysis of parameters for a given model output.

    The result of the sensitivity analysis is presented as a
    dictionary with S1, ST, S1_conf, ST_conf and names
    (described in https://salib.readthedocs.io/en/latest/api.html)
    for each evaluation performed with a sample, i.e. a polynomial
    is evaluated with a set of sample coefficients for 10 values of ``x``
    ie. ``{0: {'S1': [0.3075(x), 0.4424(y), 4.531e-27(z)], ...,
    'names': ['x', 'y', 'z']}, 1: ...}``
    """
    # pylint: disable=too-many-branches

    if len(parameters) == 0:
        raise ValueError("Cannot study the sensitivity of no variables")
    records = []
    for transmitter_map in model_output.values():
        if len(transmitter_map) > 1:
            raise ValueError("Cannot analyze sensitivity with multiple outputs")
        if len(transmitter_map) < 1:
            raise ValueError("Cannot analyze sensitivity with no output")
        for transmitter in transmitter_map.values():
            records.append(get_event_loop().run_until_complete(transmitter.load()))

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

    assert (
        len(set(record.record_type for record in records)) == 1
    ), "Bug: Requires homogeneous model output records"
    assert records[0].record_type in (
        RecordType.LIST_FLOAT,
        RecordType.SCALAR_FLOAT,
    ), "Bug: Model output must be scalar or lists"

    if records[0].record_type in (RecordType.LIST_FLOAT):
        record_size = len(records[0].data)  # type: ignore
        data = np.zeros([sample_size * param_size, record_size])
        for i, record in enumerate(records):
            for j in range(record_size):
                data[i][j] = record.data[j]  # type: ignore
    elif records[0].record_type in (RecordType.SCALAR_FLOAT):
        record_size = 1
        data = np.zeros([sample_size * param_size, record_size])
        for i, record in enumerate(records):
            data[i][0] = record.data

    problem = _build_salib_problem(parameters)

    analysis = {}
    for j in range(record_size):
        analysis[j] = fast.analyze(problem, data[:, j], M=harmonics)

    return analysis
