from ert_data import loader
from ert_data.measured import MeasuredData
from ert_shared.dark_storage.enkf import get_res
from typing import List, Union
import pandas as pd

from res.enkf import EnkfObservationImplementationType


def ensemble_parameter_names(ensemble_name: str) -> List[str]:
    res = get_res()
    return res.gen_kw_keys()


def ensemble_parameters(ensemble_name: str) -> List[dict]:
    return [dict(name=key) for key in ensemble_parameter_names(ensemble_name)]


def get_response_names():
    res = get_res()
    result = res.get_summary_keys().copy()
    result.extend(res.get_gen_data_keys().copy())
    return result


def get_responses(ensemble_name: str):
    res = get_res()
    response_names = get_response_names()
    responses = []
    active_realizations = res.get_active_realizations(ensemble_name)

    for real_id in active_realizations:
        for response_name in response_names:
            responses.append({"name": response_name, "real_id": real_id})
    return responses


def data_for_key(case, key, realization_index=None):
    """Returns a pandas DataFrame with the datapoints for a given key for a given case. The row index is
    the realization number, and the columns are an index over the indexes/dates"""

    res = get_res()
    if key.startswith("LOG10_"):
        key = key[6:]

    if res.is_summary_key(key):
        data = res.gather_summary_data(case, key, realization_index).T
    elif res.is_gen_kw_key(key):
        data = res.gather_gen_kw_data(case, key, realization_index)
        data.columns = pd.Index([0])
    elif res.is_gen_data_key(key):
        data = res.gather_gen_data_data(case, key, realization_index).T
    else:
        raise ValueError("no such key {}".format(key))

    try:
        return data.astype(float)
    except ValueError:
        return data


def observations_for_obs_keys(case, obs_keys):
    """Returns a pandas DataFrame with the datapoints for a given observation key for a given case. The row index
    is the realization number, and the column index is a multi-index with (obs_key, index/date, obs_index),
    where index/date is used to relate the observation to the data point it relates to, and obs_index is
    the index for the observation itself"""
    res = get_res()
    try:
        measured_data = MeasuredData(res, obs_keys, case_name=case, load_data=False)
        data = measured_data.data
    except loader.ObservationError:
        data = pd.DataFrame()
    expected_keys = ["OBS", "STD"]
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Invalid type: {}, should be type: {}".format(type(data), pd.DataFrame)
        )
    elif data.empty:
        return []
    elif not data.empty and not set(expected_keys).issubset(data.index):
        raise ValueError(
            "{} should be present in DataFrame index, missing: {}".format(
                ["OBS", "STD"], set(expected_keys) - set(data.index)
            )
        )
    else:
        observation_vectors = res.get_observations()
        observations = data.loc[["OBS", "STD"]]
        grouped_obs = {}
        response_observation_link = {}
        summary_obs_keys = observation_vectors.getTypedKeylist(
            EnkfObservationImplementationType.SUMMARY_OBS
        )

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
        for key, obs in grouped_obs.items():
            x_axis, values, error = (
                list(t)
                for t in zip(*sorted(zip(obs["x_axis"], obs["values"], obs["errors"])))
            )
            x_axis = _prepare_x_axis(x_axis)
            grouped_obs[key]["x_axis"] = x_axis
            grouped_obs[key]["values"] = values
            grouped_obs[key]["errors"] = error
        return [obs for obs in grouped_obs.values()]


def _get_obs_data(key, obs) -> dict:
    return dict(
        name=key,
        x_axis=obs.columns.get_level_values(0).to_list(),
        values=obs.loc["OBS"].to_list(),
        errors=obs.loc["STD"].to_list(),
    )


def _prepare_x_axis(x_axis: List[Union[int, float, str, pd.Timestamp]]) -> List[str]:
    if isinstance(x_axis[0], pd.Timestamp):
        return [pd.Timestamp(x).isoformat() for x in x_axis]

    return [str(x) for x in x_axis]
