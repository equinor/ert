from __future__ import annotations

from typing import Dict, Mapping, Union

from ert.config.gen_kw_config import GenKwConfig
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
) -> Mapping[str, Dict[str, Union[str, float]]]:
    priors_dict = {}

    for group, priors in experiment.parameter_configuration.items():
        if isinstance(priors, GenKwConfig):
            for func in priors.transform_functions:
                prior: Dict[str, Union[str, float]] = {
                    "function": _PRIOR_NAME_MAP[func.transform_function_name],
                }
                for name, value in func.parameter_list.items():
                    # Libres calls it steps, but normal stats uses bins
                    if name == "STEPS":
                        name = "bins"
                    prior[name.lower()] = value

                priors_dict[f"{group}:{func.name}"] = prior

    return priors_dict
