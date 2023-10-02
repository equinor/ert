from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Union

from ert.dark_storage.common import observations_for_obs_keys

if TYPE_CHECKING:
    from ert import LibresFacade

logger = logging.getLogger()


def create_observations(ert: LibresFacade) -> List[Dict[str, Dict[str, Any]]]:
    keys = [i.observation_key for i in ert.get_observations()]
    return observations_for_obs_keys(ert, keys)


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


def create_priors(ert: LibresFacade) -> Mapping[str, Dict[str, Union[str, float]]]:
    priors = {}
    for group, gen_kw_priors in ert.gen_kw_priors().items():
        for gen_kw_prior in gen_kw_priors:
            prior: Dict[str, Union[str, float]] = {
                "function": _PRIOR_NAME_MAP[gen_kw_prior["function"]],
            }
            for arg_name, arg_value in gen_kw_prior["parameters"].items():
                # Libres calls it steps, but normal stats uses bins
                if arg_name == "STEPS":
                    arg_name = "bins"
                prior[arg_name.lower()] = arg_value

            priors[f"{group}:{gen_kw_prior['key']}"] = prior
    return priors
