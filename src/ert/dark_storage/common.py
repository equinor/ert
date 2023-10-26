from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from ert.config import EnkfObservationImplementationType
from ert.libres_facade import LibresFacade
from ert.storage import EnsembleReader


def ensemble_parameter_names(res: LibresFacade) -> List[str]:
    return res.gen_kw_keys()


def ensemble_parameters(res: LibresFacade) -> List[Dict[str, Any]]:
    return [
        {"name": key, "userdata": {"data_origin": "GEN_KW"}, "labels": []}
        for key in ensemble_parameter_names(res)
    ]


def get_response_names(res: LibresFacade, ensemble: EnsembleReader) -> List[str]:
    result = ensemble.get_summary_keyset()
    result.extend(res.get_gen_data_keys().copy())
    return result


def data_for_key(
    res: LibresFacade,
    ensemble: EnsembleReader,
    key: str,
    realization_index: Optional[int] = None,
) -> pd.DataFrame:
    """Returns a pandas DataFrame with the datapoints for a given key for a
    given case. The row index is the realization number, and the columns are an
    index over the indexes/dates"""

    if key.split(":")[0][-1] == "H":
        return res.history_data(key, ensemble).T

    if key.startswith("LOG10_"):
        key = key[6:]
    if key in ensemble.get_summary_keyset():
        data = res.load_all_summary_data(ensemble, [key], realization_index)
        data = data[key].unstack(level="Date")
    elif key in res.gen_kw_keys():
        data = res.gather_gen_kw_data(ensemble, key, realization_index)
        if data.empty:
            return data
        data.columns = pd.Index([0])
    elif key in res.get_gen_data_keys():
        key_parts = key.split("@")
        key = key_parts[0]
        report_step = int(key_parts[1]) if len(key_parts) > 1 else 0

        try:
            data = res.load_gen_data(
                ensemble,
                key,
                report_step,
                realization_index,
            ).T
        except (ValueError, KeyError):
            data = pd.DataFrame()
    else:
        return pd.DataFrame()

    try:
        return data.astype(float)
    except ValueError:
        return data


def observations_for_obs_keys(
    res: LibresFacade, obs_keys: List[str]
) -> List[Dict[str, Any]]:
    """Returns a pandas DataFrame with the datapoints for a given observation
    key for a given case. The row index is the realization number, and the
    column index is a multi-index with (obs_key, index/date, obs_index), where
    index/date is used to relate the observation to the data point it relates
    to, and obs_index is the index for the observation itself"""
    observations = []
    for key in obs_keys:
        observation = res.config.observations[key]
        obs = {
            "name": key,
            "values": list(observation.observations.values.flatten()),
            "errors": list(observation["std"].values.flatten()),
        }
        if "time" in observation.coords:
            obs["x_axis"] = _prepare_x_axis(observation.time.values.flatten())
        else:
            obs["x_axis"] = _prepare_x_axis(
                observation["index"].values.flatten(),  # type: ignore
            )

        observations.append(obs)

    return observations


def get_observation_name(res: LibresFacade, obs_keys: List[str]) -> Optional[str]:
    summary_obs = res.get_observations().getTypedKeylist(
        EnkfObservationImplementationType.SUMMARY_OBS
    )
    for key in obs_keys:
        observation = res.config.observations[key]
        if key in summary_obs:
            return observation.name.values.flatten()[0]
        return key
    return None


def _prepare_x_axis(
    x_axis: Sequence[Union[int, float, str, pd.Timestamp]]
) -> List[str]:
    """Converts the elements of x_axis of an observation to a string suitable
    for json. If the elements are timestamps, convert to ISO-8601 format.

    >>> _prepare_x_axis([1, 2, 3, 4])
    ['1', '2', '3', '4']
    >>> _prepare_x_axis([pd.Timestamp(x, unit="d") for x in range(3)])
    ['1970-01-01T00:00:00', '1970-01-02T00:00:00', '1970-01-03T00:00:00']
    """
    if isinstance(x_axis[0], pd.Timestamp):
        return [pd.Timestamp(x).isoformat() for x in x_axis]

    return [str(x) for x in x_axis]
