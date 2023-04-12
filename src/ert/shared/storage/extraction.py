import logging
from typing import Any, List, Mapping, Union

import pandas as pd

from ert._c_wrappers.enkf.enums import EnkfObservationImplementationType
from ert.data import MeasuredData

logger = logging.getLogger()


def _prepare_x_axis(x_axis: List[Union[int, float, str, pd.Timestamp]]) -> List[str]:
    if isinstance(x_axis[0], pd.Timestamp):
        return [pd.Timestamp(x).isoformat() for x in x_axis]

    return [str(x) for x in x_axis]


def _get_obs_data(key, obs) -> Mapping[str, Any]:
    return {
        "name": key,
        "x_axis": obs.columns.get_level_values(0).to_list(),
        "values": obs.loc["OBS"].to_list(),
        "errors": obs.loc["STD"].to_list(),
    }


def create_observations(ert) -> List[Mapping[str, dict]]:
    observation_vectors = ert.get_observations()
    keys = [ert.get_observation_key(i) for i, _ in enumerate(observation_vectors)]
    summary_obs_keys = observation_vectors.getTypedKeylist(
        EnkfObservationImplementationType.SUMMARY_OBS
    )
    if keys == []:
        return []

    data = MeasuredData(ert, None, keys, load_data=False)
    observations = data.data.loc[["OBS", "STD"]]
    grouped_obs = {}
    response_observation_link = {}

    for obs_key in observations.columns.get_level_values(0).unique():
        obs_vec = observation_vectors[obs_key]
        data_key = obs_vec.getDataKey()
        obs_data = _get_obs_data(obs_key, observations[obs_key])

        if obs_key not in summary_obs_keys:
            grouped_obs[obs_key] = obs_data
            response_observation_link[data_key] = obs_key
        else:
            response_observation_link[data_key] = data_key
            if data_key in grouped_obs:
                for el in filter(lambda x: not x == "name", obs_data):
                    grouped_obs[data_key][el] += obs_data[el]
            else:
                obs_data["name"] = data_key
                grouped_obs[data_key] = obs_data
    for obs in grouped_obs.values():
        x_axis, values, error = (
            list(t)
            for t in zip(*sorted(zip(obs["x_axis"], obs["values"], obs["errors"])))
        )
        x_axis = _prepare_x_axis(x_axis)
        obs["x_axis"] = x_axis
        obs["values"] = values
        obs["errors"] = error
    return list(grouped_obs.values())


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
