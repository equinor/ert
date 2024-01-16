from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

import pandas as pd

from ert.config import EnkfObservationImplementationType
from ert.libres_facade import LibresFacade
from ert.storage import EnsembleReader, StorageReader


def ensemble_parameter_names(storage: StorageReader, ensemble_id: UUID) -> List[str]:
    return storage.get_ensemble(ensemble_id).get_gen_kw_keyset()


def ensemble_parameters(
    storage: StorageReader, ensemble_id: UUID
) -> List[Dict[str, Any]]:
    return [
        {"name": key, "userdata": {"data_origin": "GEN_KW"}, "labels": []}
        for key in ensemble_parameter_names(storage, ensemble_id)
    ]


def get_response_names(ensemble: EnsembleReader) -> List[str]:
    result = ensemble.get_summary_keyset()
    result.extend(ensemble.get_gen_data_keyset().copy())
    return result


def data_for_key(
    ensemble: EnsembleReader,
    key: str,
) -> pd.DataFrame:
    """Returns a pandas DataFrame with the datapoints for a given key for a
    given case. The row index is the realization number, and the columns are an
    index over the indexes/dates"""

    if key.startswith("LOG10_"):
        key = key[6:]
    if key in ensemble.get_summary_keyset():
        data = ensemble.load_summary(key)
        data = data[key].unstack(level="Date")
    elif key in ensemble.get_gen_kw_keyset():
        data = ensemble.load_all_gen_kw_data(key.split(":")[0])
        if data.empty:
            return pd.DataFrame()
        data = data[key].to_frame().dropna()
        data.columns = pd.Index([0])
    elif key in ensemble.get_gen_data_keyset():
        key_parts = key.split("@")
        key = key_parts[0]
        report_step = int(key_parts[1]) if len(key_parts) > 1 else 0

        try:
            data = ensemble.load_gen_data(
                key,
                report_step,
            ).T
        except (ValueError, KeyError):
            return pd.DataFrame()
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
