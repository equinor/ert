import contextlib
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
from uuid import UUID

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel

from ert.config import GenDataConfig, GenKwConfig
from ert.storage import Ensemble, Experiment, Storage


def _ensemble_parameter_names(ensemble: Ensemble) -> Iterator[str]:
    return (
        (
            f"LOG10_{config.name}:{keyword}"
            if config.shouldUseLogScale(keyword)
            else f"{config.name}:{keyword}"
        )
        for config in ensemble.experiment.parameter_configuration.values()
        if isinstance(config, GenKwConfig)
        for keyword in (e.name for e in config.transfer_functions)
    )


def ensemble_parameters(storage: Storage, ensemble_id: UUID) -> List[Dict[str, Any]]:
    return [
        {"name": key, "userdata": {"data_origin": "GEN_KW"}, "labels": []}
        for key in _ensemble_parameter_names(storage.get_ensemble(ensemble_id))
    ]


def get_response_names(ensemble: Ensemble) -> List[str]:
    result = ensemble.get_summary_keyset()
    result.extend(sorted(gen_data_keys(ensemble), key=lambda k: k.lower()))
    return result


def gen_data_keys(ensemble: Ensemble) -> Iterator[str]:
    for k, v in ensemble.experiment.response_configuration.items():
        if isinstance(v, GenDataConfig):
            if v.report_steps is None:
                yield f"{k}@0"
            else:
                for report_step in v.report_steps:
                    yield f"{k}@{report_step}"


def data_for_key(
    ensemble: Ensemble,
    key: str,
) -> pd.DataFrame:
    """Returns a pandas DataFrame with the datapoints for a given key for a
    given ensemble. The row index is the realization number, and the columns are an
    index over the indexes/dates"""
    if key.startswith("LOG10_"):
        key = key[6:]

    if key in ensemble.get_summary_keyset():
        summary_data = ensemble.load_responses(
            "summary", tuple(ensemble.get_realization_list_with_responses("summary"))
        )
        df = summary_data.to_dataframe()
        df = df.xs(key, level="name")
        df.index = df.index.rename(
            {"time": "Date", "realization": "Realization"}
        ).reorder_levels(["Realization", "Date"])
        data = df.unstack(level="Date")
        data.columns = data.columns.droplevel(0)
        try:
            return data.astype(float)
        except ValueError:
            return data

    group = key.split(":")[0]
    parameters = ensemble.experiment.parameter_configuration
    if group in parameters and isinstance(gen_kw := parameters[group], GenKwConfig):
        dataframes = []

        with contextlib.suppress(KeyError):
            try:
                data = ensemble.load_parameters(group)
            except ValueError as err:
                print(f"Could not load parameter {group}: {err}")
                return pd.DataFrame()

            da = data["transformed_values"]
            assert isinstance(da, xr.DataArray)
            da["names"] = np.char.add(f"{gen_kw.name}:", da["names"].astype(np.str_))
            df = da.to_dataframe().unstack(level="names")
            df.columns = df.columns.droplevel()
            for parameter in df.columns:
                if gen_kw.shouldUseLogScale(parameter.split(":")[1]):
                    df[f"LOG10_{parameter}"] = np.log10(df[parameter])
            dataframes.append(df)
        if not dataframes:
            return pd.DataFrame()

        dataframe = pd.concat(dataframes, axis=1)
        dataframe.columns.name = None
        dataframe.index.name = "Realization"

        data = dataframe.sort_index(axis=1)
        if data.empty:
            return pd.DataFrame()
        data = data[key].to_frame().dropna()
        data.columns = pd.Index([0])
        try:
            return data.astype(float)
        except ValueError:
            return data
    if key in gen_data_keys(ensemble):
        key_parts = key.split("@")
        key = key_parts[0]
        try:
            mask = ensemble.get_realization_mask_with_responses(key)
            realizations = np.where(mask)[0]
            data = ensemble.load_responses(key, tuple(realizations))
        except ValueError as err:
            print(f"Could not load response {key}: {err}")
            return pd.DataFrame()

        report_step = int(key_parts[1]) if len(key_parts) > 1 else 0

        try:
            vals = data.sel(report_step=report_step, drop=True)
            index = pd.Index(vals.index.values, name="axis")
            data = pd.DataFrame(
                data=vals["values"].values.reshape(len(vals.realization), -1).T,
                index=index,
                columns=realizations,
            ).T
            try:
                return data.astype(float)
            except ValueError:
                return data
        except (ValueError, KeyError):
            return pd.DataFrame()

    return pd.DataFrame()


class _ObservationWithXAxis(BaseModel):
    obs_name: str
    values: List[float]
    errors: List[float]
    x_axis: List[str]


def get_observation_for_response(
    ensemble: Ensemble, response_key: str
) -> Optional[_ObservationWithXAxis]:
    if "@" in response_key:
        [response_key, _] = response_key.split("@")

    for dataset in ensemble.experiment.observations.values():
        if response_key in dataset["name"]:
            x_coord_key = "time" if "time" in dataset.coords else "index"
            ds = dataset.sel(name=response_key)
            df = ds.to_dataframe().dropna().reset_index()
            return _ObservationWithXAxis(
                obs_name=df["obs_name"].to_list()[0],
                values=df["observations"].to_list(),
                errors=df["std"].to_list(),
                x_axis=_prepare_x_axis(df[x_coord_key].to_list()),
            )

    return None


def get_all_observations(experiment: Experiment) -> List[Dict[str, Any]]:
    observations = []
    for dataset in experiment.observations.values():
        x_coord_key = "time" if "time" in dataset.coords else "index"

        for obs_name in dataset["obs_name"].values.flatten():
            ds = dataset.sel(obs_name=obs_name)
            df = ds.to_dataframe().reset_index()
            observations.append(
                {
                    "name": obs_name,
                    "values": df["observations"].to_list(),
                    "errors": df["std"].to_list(),
                    "x_axis": _prepare_x_axis(df[x_coord_key].to_list()),
                }
            )

    return observations


def get_observations_for_obs_keys(
    ensemble: Ensemble, observation_keys: List[str]
) -> List[Dict[str, Any]]:
    observations = []
    experiment_observations = ensemble.experiment.observations

    for ds in experiment_observations.values():
        for obs_key, obs_ds in ds.groupby("obs_name"):
            if obs_key not in observation_keys:
                continue

            df = obs_ds.to_dataframe().reset_index()
            observation = {
                "name": obs_key,
                "values": list(df["observations"].to_list()),
                "errors": list(df["std"].to_list()),
            }
            if "time" in obs_ds.coords:
                observation["x_axis"] = _prepare_x_axis(df["time"].to_list())
            else:
                observation["x_axis"] = _prepare_x_axis(df["index"].to_list())
            observations.append(observation)

    return observations


def get_observation_name(ensemble: Ensemble, observation_keys: List[str]) -> str:
    observations_dict = ensemble.experiment.observations
    for key in observation_keys:
        observation = observations_dict[key]
        if observation.response == "summary":
            return observation.name.values.flatten()[0]
        return key
    return ""


def get_observation_keys_for_response(
    ensemble: Ensemble, response_key: str
) -> List[str]:
    """
    Get all observation keys for given response key
    """

    if response_key in gen_data_keys(ensemble):
        response_key = response_key.split("@")[0]

    return ensemble.experiment.observations_for_response(response_key)["obs_name"].data


def _prepare_x_axis(
    x_axis: Sequence[Union[int, float, str, pd.Timestamp]],
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
