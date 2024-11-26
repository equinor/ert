"""
Read-only API to fetch responses (a.k.a measurements) and
matching observations from internal ERT-storage.
The main goal is to facilitate data-analysis using scipy and similar tools,
instead of having to implement analysis-functionality into ERT using C/C++.
The API is typically meant used as part of workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd
import polars

if TYPE_CHECKING:
    from ert.storage import Ensemble


class ResponseError(Exception):
    pass


class MeasuredData:
    def __init__(
        self,
        ensemble: Ensemble,
        keys: Optional[List[str]] = None,
    ):
        if keys is None:
            keys = sorted(ensemble.experiment.observation_keys)
        if not keys:
            raise ObservationError("No observation keys provided")

        self._set_data(self._get_data(ensemble, keys))

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def _set_data(self, data: pd.DataFrame) -> None:
        expected_keys = {"OBS", "STD"}
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Invalid type: {type(data)}, should be type: {pd.DataFrame}"
            )
        if not expected_keys.issubset(data.index):
            missing = expected_keys - set(data.index)
            raise ValueError(
                f"{expected_keys} should be present in DataFrame index, \
                missing: {missing}"
            )
        self._data = data

    def remove_failed_realizations(self) -> None:
        """Removes rows with no simulated data, leaving observations and
        standard deviations as-is."""
        pre_index = self.data.index
        post_index = list(self.data.dropna(axis=0, how="all").index)
        drop_index = set(pre_index) - {*post_index, "STD", "OBS"}
        self._set_data(self.data.drop(index=drop_index))

    def get_simulated_data(self) -> pd.DataFrame:
        """Dimension of data is (number of responses x number of realizations)."""
        return self.data[~self.data.index.isin(["OBS", "STD"])]

    def remove_inactive_observations(self) -> None:
        """Removes columns with one or more NaN or inf values."""
        filtered_dataset = self.data.replace([np.inf, -np.inf], np.nan).dropna(
            axis="columns", how="any"
        )
        if filtered_dataset.empty:
            raise ValueError(
                "This operation results in an empty dataset "
                "(could be due to one or more failed realizations)"
            )
        self._set_data(filtered_dataset)

    def is_empty(self) -> bool:
        return bool(self.data.empty)

    @staticmethod
    def _get_data(
        ensemble: Ensemble,
        observation_keys: List[str],
    ) -> pd.DataFrame:
        """
        Adds simulated and observed data and returns a dataframe where ensemble
        members will have a data key, observed data will be named OBS and
        observed standard deviation will be named STD.
        """

        observations_by_type = ensemble.experiment.observations

        dfs = []

        for key in observation_keys:
            if key not in ensemble.experiment.observation_keys:
                raise ObservationError(
                    f"No observation: {key} in ensemble: {ensemble.name}"
                )

        for (
            response_type,
            response_cls,
        ) in ensemble.experiment.response_configuration.items():
            observations_for_type = observations_by_type[response_type].filter(
                polars.col("observation_key").is_in(observation_keys)
            )
            responses_for_type = ensemble.load_responses(
                response_type,
                realizations=tuple(ensemble.get_realization_list_with_responses()),
            )

            if responses_for_type.is_empty():
                raise ResponseError(
                    f"No response loaded for observation type: {response_type}"
                )

            # Note that if there are duplicate entries for one
            # response at one index, they are aggregated together
            # with "mean" by default
            pivoted = responses_for_type.pivot(
                on="realization",
                index=["response_key", *response_cls.primary_key],
                aggregate_function="mean",
            )

            if "time" in pivoted:
                joined = observations_for_type.join_asof(
                    pivoted,
                    by=["response_key", *response_cls.primary_key],
                    on="time",
                    tolerance="1s",
                )
            else:
                joined = observations_for_type.join(
                    pivoted,
                    how="left",
                    on=["response_key", *response_cls.primary_key],
                )

            joined = joined.sort(by="observation_key").with_columns(
                polars.concat_str(response_cls.primary_key, separator=", ").alias(
                    "key_index"
                )
            )

            # Put key_index column 1st
            joined = joined[["key_index", *joined.columns[:-1]]]
            joined = joined.drop(*response_cls.primary_key)

            if not joined.is_empty():
                dfs.append(joined)

        df = polars.concat(dfs)
        df = df.rename(
            {
                "observations": "OBS",
                "std": "STD",
            }
        )

        pddf = df.to_pandas()[
            [
                "observation_key",
                "key_index",
                "OBS",
                "STD",
                *df.columns[5:],
            ]
        ]

        # Pandas differentiates vs int and str keys.
        # Legacy-wise we use int keys for realizations
        pddf.rename(
            columns={str(k): int(k) for k in range(ensemble.ensemble_size)},
            inplace=True,
        )

        pddf = pddf.set_index(["observation_key", "key_index"]).transpose()

        return pddf


class ObservationError(Exception):
    pass
