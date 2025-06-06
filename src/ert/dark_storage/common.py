import logging
import os
from collections.abc import Callable
from typing import Any

import pandas as pd
import polars as pl
from polars.exceptions import ColumnNotFoundError

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


# indexing below is based on observation ds columns:
# [ "observation_key", "response_key", *primary_key ]
# for gen_data primary_key is ["report_step", "index"]
# for summary it is ["time"]
response_to_pandas_x_axis_fns: dict[str, Callable[[tuple[Any, ...]], Any]] = {
    "summary": lambda t: pd.Timestamp(t[2]).isoformat(),
    "gen_data": lambda t: str(t[3]),
}


def _extract_parameter_group_and_key(key: str) -> tuple[str, str] | tuple[None, None]:
    key = key.removeprefix("LOG10_")
    if ":" not in key:
        # Assume all incoming keys are in format group:key for now
        return None, None

    param_group, param_key = key.split(":")
    return param_group, param_key


def data_for_parameter(ensemble: Ensemble, key: str) -> pd.DataFrame:
    group, _ = _extract_parameter_group_and_key(key)
    try:
        df = ensemble.load_scalars(group)
    except KeyError:
        return pd.DataFrame()

    if df.is_empty():
        return pd.DataFrame()

    dataframe = df.to_pandas().set_index("realization")
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


def data_for_response(
    ensemble: Ensemble, key: str, filter_on: dict[str, Any] | None = None
) -> pd.DataFrame:
    response_key, response_type = _extract_response_type_and_key(
        key, ensemble.experiment.response_key_to_response_type
    )

    if response_key is None:
        return pd.DataFrame()

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
            data = ensemble.load_responses(
                response_key, tuple(ensemble.get_realization_list_with_responses())
            )
        except ValueError as err:
            logger.info(f"Dark storage could not load response {key}: {err}")
            return pd.DataFrame()

        try:
            assert filter_on is not None
            assert "report_step" in filter_on
            report_step = int(filter_on["report_step"])
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
