import logging
from typing import List, Mapping, Union

import pandas as pd

from ert.dark_storage.common import observations_for_obs_keys

logger = logging.getLogger()


def _prepare_x_axis(x_axis: List[Union[int, float, str, pd.Timestamp]]) -> List[str]:
    if isinstance(x_axis[0], pd.Timestamp):
        return [pd.Timestamp(x).isoformat() for x in x_axis]

    return [str(x) for x in x_axis]


def create_observations(ert) -> List[Mapping[str, dict]]:
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


def create_priors(ert) -> Mapping[str, dict]:
    priors = {}
    for group, gen_kw_priors in ert.gen_kw_priors().items():
        for gen_kw_prior in gen_kw_priors:
            prior = {
                "function": _PRIOR_NAME_MAP[gen_kw_prior["function"]],
            }
            for arg_name, arg_value in gen_kw_prior["parameters"].items():
                # triangular uses X<arg_name>, removing the x prefix
                if arg_name.startswith("X"):
                    arg_name = arg_name[1:]
                # Libres calls it steps, but normal stats uses bins
                if arg_name == "STEPS":
                    arg_name = "bins"
                prior[arg_name.lower()] = arg_value

            priors[f"{group}:{gen_kw_prior['key']}"] = prior
    return priors
