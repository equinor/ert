import contextlib
import logging
import operator
import os
from collections.abc import Callable, Iterator
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from polars.exceptions import ColumnNotFoundError

from ert.config import Field, GenDataConfig, GenKwConfig
from ert.dark_storage.exceptions import InternalServerError
from ert.storage import Ensemble, Experiment, Storage, open_storage

logger = logging.getLogger(__name__)


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        try:
            return (_storage := open_storage(os.environ["ERT_STORAGE_ENS_PATH"]))
        except RuntimeError as err:
            raise InternalServerError(f"{err!s}") from err
    _storage.refresh()
    return _storage


response_key_to_displayed_key: dict[str, Callable[[tuple[Any, ...]], str]] = {
    "summary": operator.itemgetter(0),
    "gen_data": lambda t: f"{t[0]}@{t[1]}",
}


def _parse_gendata_response_key(display_key: str) -> tuple[Any, ...]:
    response_key, report_step = display_key.split("@")
    return response_key, int(report_step)


displayed_key_to_response_key: dict[str, Callable[[str], tuple[Any, ...]]] = {
    "summary": lambda key: (key,),
    "gen_data": _parse_gendata_response_key,
}

# indexing below is based on observation ds columns:
# [ "observation_key", "response_key", *primary_key ]
# for gen_data primary_key is ["report_step", "index"]
# for summary it is ["time"]
response_to_pandas_x_axis_fns: dict[str, Callable[[tuple[Any, ...]], Any]] = {
    "summary": lambda t: pd.Timestamp(t[2]).isoformat(),
    "gen_data": lambda t: str(t[3]),
}


def ensemble_parameters(storage: Storage, ensemble_id: UUID) -> list[dict[str, Any]]:
    param_list = []
    ensemble = storage.get_ensemble(ensemble_id)
    for config in ensemble.experiment.parameter_configuration.values():
        match config:
            case GenKwConfig(name=name, transform_functions=transform_functions):
                for tf in transform_functions:
                    param_list.append(
                        {
                            "name": (
                                f"LOG10_{name}:{tf.name}"
                                if tf.use_log
                                else f"{name}:{tf.name}"
                            ),
                            "userdata": {"data_origin": "GEN_KW"},
                            "dimensionality": 1,
                            "labels": [],
                        }
                    )
            case Field(name=name, nx=nx, ny=ny, nz=nz):
                param_list.append(
                    {
                        "name": name,
                        "userdata": {
                            "data_origin": "FIELD",
                            "nx": nx,
                            "ny": ny,
                            "nz": nz,
                        },
                        "dimensionality": 3,
                        "labels": [],
                    }
                )

    return param_list


def get_response_names(ensemble: Ensemble) -> list[str]:
    result = ensemble.experiment.response_type_to_response_keys["summary"]
    result.extend(sorted(gen_data_display_keys(ensemble), key=lambda k: k.lower()))
    return result


def gen_data_display_keys(ensemble: Ensemble) -> Iterator[str]:
    gen_data_config = ensemble.experiment.response_configuration.get("gen_data")

    if gen_data_config:
        assert isinstance(gen_data_config, GenDataConfig)
        for key, report_steps in zip(
            gen_data_config.keys, gen_data_config.report_steps_list, strict=False
        ):
            if report_steps is None:
                yield f"{key}@0"
            else:
                for report_step in report_steps:
                    yield f"{key}@{report_step}"


def _extract_parameter_group_and_key(key: str) -> tuple[str, str] | tuple[None, None]:
    key = key.removeprefix("LOG10_")
    if ":" not in key:
        # Assume all incoming keys are in format group:key for now
        return None, None

    param_group, param_key = key.split(":")
    return param_group, param_key


def data_for_parameter(ensemble: Ensemble, key: str) -> pd.DataFrame:
    group, _ = _extract_parameter_group_and_key(key)
    parameters = ensemble.experiment.parameter_configuration
    if group in parameters and isinstance(gen_kw := parameters[group], GenKwConfig):
        dataframes: list[pd.DataFrame] = []

        with contextlib.suppress(KeyError):
            try:
                da = ensemble.load_parameters(group, transformed=True)["values"]
            except ValueError as err:
                print(f"Could not load parameter {group}: {err}")
                return pd.DataFrame()

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
        if data.empty or key not in data:
            return pd.DataFrame()
        data = data[key].to_frame().dropna()
        data.columns = pd.Index([0])
        try:
            return data.astype(float)
        except ValueError:
            return data


def _extract_response_type_and_key(
    key: str, response_key_to_response_type: dict[str, str]
) -> tuple[str, str] | tuple[None, None]:
    # Check for exact match first. For example if key is "FOPRH"
    # it may stop at "FOPR", which would be incorrect
    response_key = next((k for k in response_key_to_response_type if k == key), None)
    if response_key is None:
        response_key = next(
            (k for k in response_key_to_response_type if k in key and key != f"{k}H"),
            None,
        )

    if response_key is None:
        return None, None

    response_type = response_key_to_response_type.get(response_key)
    assert response_type is not None

    return response_key, response_type


def data_for_response(ensemble: Ensemble, key: str) -> pd.DataFrame:
    response_key, response_type = _extract_response_type_and_key(
        key, ensemble.experiment.response_key_to_response_type
    )

    assert response_key is not None
    assert response_type is not None

    if response_type == "summary":
        summary_data = ensemble.load_responses(
            response_key,
            tuple(ensemble.get_realization_list_with_responses()),
        )
        if summary_data.is_empty():
            return pd.DataFrame()

        df = (
            summary_data.rename({"time": "Date", "realization": "Realization"})
            .drop("response_key")
            .to_pandas()
        )
        df = df.set_index(["Date", "Realization"])
        # This performs the same aggragation by mean of duplicate values
        # as in ert/analysis/_es_update.py
        df = df.groupby(["Date", "Realization"]).mean()
        data = df.unstack(level="Date")
        data.columns = data.columns.droplevel(0)
        try:
            return data.astype(float)
        except ValueError:
            return data

    if response_type == "gen_data":
        try:
            # Call below will ValueError if key ends with H,
            # requested via PlotAPI.history_data
            response_key, report_step = displayed_key_to_response_key["gen_data"](key)
            assert isinstance(response_key, str)
            data = ensemble.load_responses(
                response_key, tuple(ensemble.get_realization_list_with_responses())
            )
        except ValueError as err:
            logger.info(f"Dark storage could not load response {key}: {err}")
            return pd.DataFrame()

        try:
            vals = data.filter(pl.col("report_step").eq(report_step))
            pivoted = vals.drop("response_key", "report_step").pivot(
                on="index", values="values"
            )
            data = pivoted.to_pandas().set_index("realization")
            data.columns = data.columns.astype(int)
            data.columns.name = "axis"
            try:
                return data.astype(float)
            except ValueError:
                return data
        except (ValueError, KeyError, ColumnNotFoundError):
            return pd.DataFrame()


def data_for_key(
    ensemble: Ensemble,
    key: str,
) -> pd.DataFrame:
    """Returns a pandas DataFrame with the datapoints for a given key for a
    given ensemble. The row index is the realization number, and the columns are an
    index over the indexes/dates"""

    response_key, _ = _extract_response_type_and_key(
        key, ensemble.experiment.response_key_to_response_type
    )

    if response_key is not None:
        return data_for_response(ensemble, key)

    parameter_config = ensemble.experiment.parameter_configuration
    parameter_group, parameter_key = _extract_parameter_group_and_key(key)
    if (
        parameter_group in parameter_config
        and parameter_key in parameter_config[parameter_group].parameter_keys
    ):
        return data_for_parameter(ensemble, key)

    return pd.DataFrame()


def _get_observations(
    experiment: Experiment, observation_keys: list[str] | None = None
) -> list[dict[str, Any]]:
    observations = []

    for response_type, df in experiment.observations.items():
        if observation_keys is not None:
            df = df.filter(pl.col("observation_key").is_in(observation_keys))

        if df.is_empty():
            continue

        x_axis_fn = response_to_pandas_x_axis_fns[response_type]
        df = df.rename(
            {
                "observation_key": "name",
                "std": "errors",
                "observations": "values",
            }
        )
        df = df.with_columns(pl.Series(name="x_axis", values=df.map_rows(x_axis_fn)))
        df = df.sort("x_axis")

        for obs_key, _obs_df in df.group_by("name"):
            observations.append(
                {
                    "name": obs_key[0],
                    "values": _obs_df["values"].to_list(),
                    "errors": _obs_df["errors"].to_list(),
                    "x_axis": _obs_df["x_axis"].to_list(),
                }
            )

    return observations


def get_all_observations(experiment: Experiment) -> list[dict[str, Any]]:
    return _get_observations(experiment)


def get_observations_for_obs_keys(
    ensemble: Ensemble, observation_keys: list[str]
) -> list[dict[str, Any]]:
    return _get_observations(ensemble.experiment, observation_keys)


def get_observation_keys_for_response(
    ensemble: Ensemble, displayed_response_key: str
) -> list[str]:
    """
    Get all observation keys for given response key
    """

    if displayed_response_key in gen_data_display_keys(ensemble):
        response_key, report_step = displayed_key_to_response_key["gen_data"](
            displayed_response_key
        )

        if "gen_data" in ensemble.experiment.observations:
            observations = ensemble.experiment.observations["gen_data"]
            filtered = observations.filter(
                pl.col("response_key").eq(response_key)
                & pl.col("report_step").eq(report_step)
            )

            if filtered.is_empty():
                return []

            return filtered["observation_key"].unique().to_list()

    elif (
        displayed_response_key
        in ensemble.experiment.response_type_to_response_keys.get("summary", {})
    ):
        response_key = displayed_key_to_response_key["summary"](displayed_response_key)[
            0
        ]

        if "summary" in ensemble.experiment.observations:
            observations = ensemble.experiment.observations["summary"]
            filtered = observations.filter(pl.col("response_key").eq(response_key))

            if filtered.is_empty():
                return []

            return filtered["observation_key"].unique().to_list()

    return []
