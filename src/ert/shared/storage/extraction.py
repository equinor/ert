from __future__ import annotations

from collections.abc import Mapping

from ert.config import GenKwConfig
from ert.storage import Experiment

_PRIOR_NAME_MAP = {
    "NORMAL": "normal",
    "LOGNORMAL": "lognormal",
    "TRIANGULAR": "trig",
    "TRUNCATED_NORMAL": "ert_truncnormal",
    "CONST": "const",
    "UNIFORM": "uniform",
    "LOGUNIF": "loguniform",
    "DUNIF": "ert_duniform",
    "RAW": "stdnormal",
    "ERRF": "ert_erf",
    "DERRF": "ert_derf",
}


def create_priors(
    experiment: Experiment,
) -> Mapping[str, dict[str, str | float]]:
    priors_dict = {}

    for group, priors in experiment.parameter_configuration.items():
        if isinstance(priors, GenKwConfig):
            for func in priors.transform_functions:
                prior: dict[str, str | float] = {
                    "function": _PRIOR_NAME_MAP[func.distribution.name.upper()],
                }
                for name, value in func.parameter_list.items():
                    # Libres calls it steps, but normal stats uses bins
                    if name == "STEPS":
                        name = "bins"
                    prior[name.lower()] = value

                priors_dict[f"{group}:{func.name}"] = prior

    return priors_dict
