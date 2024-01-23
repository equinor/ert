from typing import Any, Dict, List, Sequence, Union
from uuid import UUID

import pandas as pd

from ert.storage import EnsembleReader, ExperimentReader, StorageReader


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


def get_all_observations(experiment: ExperimentReader) -> List[Dict[str, Any]]:
    observations = []
    for key, dataset in experiment.observations.items():
        observation = {
            "name": key,
            "values": list(dataset["observations"].values.flatten()),
            "errors": list(dataset["std"].values.flatten()),
        }
        if "time" in dataset.coords:
            observation["x_axis"] = _prepare_x_axis(dataset["time"].values.flatten())  # type: ignore
        else:
            observation["x_axis"] = _prepare_x_axis(dataset["index"].values.flatten())  # type: ignore
        observations.append(observation)

    observations.sort(key=lambda x: x["x_axis"])  # type: ignore
    return observations


def get_observations_for_obs_keys(
    ensemble: EnsembleReader, observation_keys: List[str]
) -> List[Dict[str, Any]]:
    observations = []
    for key in observation_keys:
        dataset = ensemble.experiment.observations[key]
        observation = {
            "name": key,
            "values": list(dataset["observations"].values.flatten()),
            "errors": list(dataset["std"].values.flatten()),
        }
        if "time" in dataset.coords:
            observation["x_axis"] = _prepare_x_axis(dataset["time"].values.flatten())  # type: ignore
        else:
            observation["x_axis"] = _prepare_x_axis(dataset["index"].values.flatten())  # type: ignore
        observations.append(observation)

    observations.sort(key=lambda x: x["x_axis"])  # type: ignore
    return observations


def get_observation_name(ensemble: EnsembleReader, observation_keys: List[str]) -> str:
    for key in observation_keys:
        observation = ensemble.experiment.observations[key]
        if observation.response == "summary":
            return observation.name.values.flatten()[0]
        return key
    return ""


def get_observation_keys_for_response(
    ensemble: EnsembleReader, response_key: str
) -> List[str]:
    """
    Get all observation keys for given response key
    """

    if response_key in ensemble.get_gen_data_keyset():
        response_key_parts = response_key.split("@")
        data_key = response_key_parts[0]
        data_report_step = (
            int(response_key_parts[1]) if len(response_key_parts) > 1 else 0
        )

        for observation_key, dataset in ensemble.experiment.observations.items():
            if (
                "report_step" in dataset.coords
                and data_key == dataset.attrs["response"]
                and data_report_step == min(dataset["report_step"].values)
            ):
                return [observation_key]
        return []

    elif response_key in ensemble.get_summary_keyset():
        observation_keys = []
        for observation_key, dataset in ensemble.experiment.observations.items():
            if (
                dataset.attrs["response"] == "summary"
                and dataset.name.values.flatten()[0] == response_key
            ):
                observation_keys.append(observation_key)
        return observation_keys

    return []


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
