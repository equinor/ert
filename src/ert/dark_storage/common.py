import contextlib
from typing import Any, Dict, Iterator, List, Sequence, Union
from uuid import UUID

import numpy as np
import pandas as pd
import xarray as xr

from ert.config import GenDataConfig, GenKwConfig
from ert.config.field import Field
from ert.storage import Ensemble, Experiment, Storage


def ensemble_parameters(storage: Storage, ensemble_id: UUID) -> List[Dict[str, Any]]:
    param_list = []
    ensemble = storage.get_ensemble(ensemble_id)
    for config in ensemble.experiment.parameter_configuration.values():
        if isinstance(config, GenKwConfig):
            for keyword in (e.name for e in config.transform_functions):
                param_list.append(
                    {
                        "name": (
                            f"LOG10_{config.name}:{keyword}"
                            if config.shouldUseLogScale(keyword)
                            else f"{config.name}:{keyword}"
                        ),
                        "userdata": {"data_origin": "GEN_KW"},
                        "dimensionality": 1,
                        "labels": [],
                    }
                )
        elif isinstance(config, Field):
            param_list.append(
                {
                    "name": config.name,
                    "userdata": {
                        "data_origin": "FIELD",
                        "nx": config.nx,
                        "ny": config.ny,
                        "nz": config.nz,
                    },
                    "dimensionality": 3,
                    "labels": [],
                }
            )

    return param_list


def get_response_names(ensemble: Ensemble) -> List[str]:
    result = ensemble.get_summary_keyset()
    result.extend(sorted(gen_data_keys(ensemble), key=lambda k: k.lower()))
    return result


def gen_data_keys(ensemble: Ensemble) -> Iterator[str]:
    gen_data_config = ensemble.experiment.response_configuration.get("gen_data")

    if gen_data_config:
        assert isinstance(gen_data_config, GenDataConfig)
        for key, report_steps in zip(
            gen_data_config.keys, gen_data_config.report_steps_list
        ):
            if report_steps is None:
                yield f"{key}@0"
            else:
                for report_step in report_steps:
                    yield f"{key}@{report_step}"


def data_for_key(
    ensemble: Ensemble,
    key: str,
) -> pd.DataFrame:
    """Returns a pandas DataFrame with the datapoints for a given key for a
    given ensemble. The row index is the realization number, and the columns are an
    index over the indexes/dates"""
    if key.startswith("LOG10_"):
        key = key[6:]

    try:
        summary_data = ensemble.load_responses(
            "summary", tuple(ensemble.get_realization_list_with_responses("summary"))
        )
        summary_keys = summary_data["name"].values
    except (ValueError, KeyError):
        summary_data = xr.Dataset()
        summary_keys = np.array([], dtype=str)

    if key in summary_keys:
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
                data=vals["values"].values.reshape(len(vals.realization), -1),
                index=realizations,
                columns=index,
            )
            try:
                return data.astype(float)
            except ValueError:
                return data
        except (ValueError, KeyError):
            return pd.DataFrame()

    return pd.DataFrame()


def get_all_observations(experiment: Experiment) -> List[Dict[str, Any]]:
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
    ensemble: Ensemble, observation_keys: List[str]
) -> List[Dict[str, Any]]:
    observations = []
    experiment_observations = ensemble.experiment.observations
    for key in observation_keys:
        dataset = experiment_observations[key]
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


def get_observation_keys_for_response(
    ensemble: Ensemble, response_key: str
) -> List[str]:
    """
    Get all observation keys for given response key
    """

    if response_key in gen_data_keys(ensemble):
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
