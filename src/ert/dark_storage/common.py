import contextlib
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from uuid import UUID

import numpy as np
import pandas as pd
import polars
import xarray as xr
from polars.exceptions import ColumnNotFoundError

from ert.config import GenDataConfig, GenKwConfig
from ert.config.field import Field
from ert.storage import Ensemble, Experiment, Storage

response_key_to_displayed_key: Dict[str, Callable[[Tuple[Any, ...]], str]] = {
    "summary": lambda t: t[0],
    "gen_data": lambda t: f"{t[0]}@{t[1]}",
}


def _parse_gendata_response_key(display_key: str) -> Tuple[Any, ...]:
    response_key, report_step = display_key.split("@")
    return response_key, int(report_step)


displayed_key_to_response_key: Dict[str, Callable[[str], Tuple[Any, ...]]] = {
    "summary": lambda key: (key,),
    "gen_data": _parse_gendata_response_key,
}

# indexing below is based on observation ds columns:
# [ "observation_key", "response_key", *primary_key ]
# for gen_data primary_key is ["report_step", "index"]
# for summary it is ["time"]
response_to_pandas_x_axis_fns: Dict[str, Callable[[Tuple[Any, ...]], Any]] = {
    "summary": lambda t: pd.Timestamp(t[2]).isoformat(),
    "gen_data": lambda t: str(t[3]),
}


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
        summary_keys = summary_data["response_key"].unique().to_list()
    except (ValueError, KeyError, polars.exceptions.ColumnNotFoundError):
        summary_data = polars.DataFrame()
        summary_keys = []

    if key in summary_keys:
        df = (
            summary_data.filter(polars.col("response_key").eq(key))
            .rename({"time": "Date", "realization": "Realization"})
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
        if data.empty or key not in data:
            return pd.DataFrame()
        data = data[key].to_frame().dropna()
        data.columns = pd.Index([0])
        try:
            return data.astype(float)
        except ValueError:
            return data
    if key in gen_data_keys(ensemble):
        response_key, report_step = displayed_key_to_response_key["gen_data"](key)
        try:
            mask = ensemble.get_realization_mask_with_responses(response_key)
            realizations = np.where(mask)[0]
            data = ensemble.load_responses(response_key, tuple(realizations))
        except ValueError as err:
            print(f"Could not load response {key}: {err}")
            return pd.DataFrame()

        try:
            vals = data.filter(polars.col("report_step").eq(report_step))
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

    return pd.DataFrame()


def _get_observations(
    experiment: Experiment, observation_keys: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    observations = []

    for response_type, df in experiment.observations.items():
        if observation_keys is not None:
            df = df.filter(polars.col("observation_key").is_in(observation_keys))

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
        df = df.with_columns(
            polars.Series(name="x_axis", values=df.map_rows(x_axis_fn))
        )
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


def get_all_observations(experiment: Experiment) -> List[Dict[str, Any]]:
    return _get_observations(experiment)


def get_observations_for_obs_keys(
    ensemble: Ensemble, observation_keys: List[str]
) -> List[Dict[str, Any]]:
    return _get_observations(ensemble.experiment, observation_keys)


def get_observation_keys_for_response(
    ensemble: Ensemble, displayed_response_key: str
) -> List[str]:
    """
    Get all observation keys for given response key
    """

    if displayed_response_key in gen_data_keys(ensemble):
        response_key, report_step = displayed_key_to_response_key["gen_data"](
            displayed_response_key
        )

        if "gen_data" in ensemble.experiment.observations:
            observations = ensemble.experiment.observations["gen_data"]
            filtered = observations.filter(
                polars.col("response_key").eq(response_key)
                & polars.col("report_step").eq(report_step)
            )

            if filtered.is_empty():
                return []

            return filtered["observation_key"].unique().to_list()

    elif displayed_response_key in ensemble.get_summary_keyset():
        response_key = displayed_key_to_response_key["summary"](displayed_response_key)[
            0
        ]

        if "summary" in ensemble.experiment.observations:
            observations = ensemble.experiment.observations["summary"]
            filtered = observations.filter(polars.col("response_key").eq(response_key))

            if filtered.is_empty():
                return []

            return filtered["observation_key"].unique().to_list()

    return []
