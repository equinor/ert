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

    for param in experiment.parameter_configuration.values():
        if isinstance(param, GenKwConfig):
            prior: dict[str, str | float] = {
                "function": _PRIOR_NAME_MAP[param.distribution.name.upper()],
                **param.distribution.model_dump(exclude={"name"}),
            }

            priors_dict[f"{param.group}:{param.name}"] = prior

    return priors_dict
