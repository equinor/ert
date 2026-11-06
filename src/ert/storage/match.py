from __future__ import annotations

import logging
from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, override

import polars as pl

from ert.config._reservoir_data_utils import SeismicData
from ert.config.breakthrough_config import BreakthroughConfig
from ert.config.gen_data_config import GenDataConfig
from ert.config.observation_quality_control import (
    ensure_qc_error_column,
    qc_rft_observations,
    qc_seismic_observations,
)
from ert.config.response_config import ResponseConfig
from ert.config.rft_config import RFTConfig
from ert.config.seismic_config import SeismicConfig
from ert.config.summary_config import SummaryConfig

if TYPE_CHECKING:
    from .local_ensemble import LocalEnsemble

logger = logging.getLogger(__name__)


@dataclass
class GenDataMatchConfig:
    pass


@dataclass
class SummaryMatchConfig:
    pass


@dataclass
class BreakthroughMatchConfig:
    fill_in_unreached_threshold: bool = False


@dataclass
class RFTMatchConfig:
    approximate_values: bool = True


@dataclass
class SeismicMatchConfig:
    pass


@dataclass
class MatchConfig:
    rft: RFTMatchConfig = field(default_factory=RFTMatchConfig)
    seismic: SeismicMatchConfig = field(default_factory=SeismicMatchConfig)
    gen_data: GenDataMatchConfig = field(default_factory=GenDataMatchConfig)
    summary: SummaryMatchConfig = field(default_factory=SummaryMatchConfig)
    breakthrough: BreakthroughMatchConfig = field(
        default_factory=BreakthroughMatchConfig
    )

    special_match_keys: list[str] | None = None
    convert_response_key_to_categorical: bool = False
    obs_keys_to_filter_observations_on: Iterable[str] | None = None
    qc_observations: bool = False
    response_key_to_filter_response_on: str | None = None
    filter_out_unmatchable_responses: bool = False
    pivot_responses_on: str | None = None


class ReservoirDataMatchStrategy[ConcreteResponseConfig: ResponseConfig](ABC):
    ensemble: LocalEnsemble
    response_cls: ConcreteResponseConfig
    match_config: MatchConfig
    match_keys: list[str]

    def __init__(
        self,
        ensemble: LocalEnsemble,
        response_cls: ConcreteResponseConfig,
        match_config: MatchConfig,
    ) -> None:
        self.ensemble = ensemble
        self.response_cls = response_cls
        self.match_config = match_config
        if self.match_config.special_match_keys is not None:
            self.match_keys = self.match_config.special_match_keys
        else:
            self.match_keys = ["response_key", *response_cls.match_key]

    def prepare_observations(self, observations: pl.DataFrame) -> pl.DataFrame:
        observations = self.customize_observations(observations)

        if self.match_config.obs_keys_to_filter_observations_on is not None:
            selected_observations = self.match_config.obs_keys_to_filter_observations_on
            observations = observations.filter(
                pl.col("observation_key").is_in(list(selected_observations))
            )

        if self.match_config.convert_response_key_to_categorical:
            observations = observations.with_columns(
                [pl.col("response_key").cast(pl.Categorical)]
            )

        return observations

    def extract_responses(
        self, realizations: tuple[int, ...], observations: pl.DataFrame
    ) -> pl.DataFrame:
        lazy_responses = self._extract_lazy_responses(realizations, observations)
        return self._collect_responses(lazy_responses, observations)

    def match(
        self,
        observations: pl.DataFrame,
        responses: pl.DataFrame,
    ) -> pl.DataFrame:
        return observations.join(
            responses,
            how="left",
            on=self.match_keys,
            nulls_equal=True,
        )

    def customize_observations(self, observations: pl.DataFrame) -> pl.DataFrame:
        return observations

    def enrich_observations_with_realization_metadata(
        self, observations: pl.DataFrame, realization: int
    ) -> pl.DataFrame:
        return observations

    def customize_lazy_responses(
        self, lazy_responses: pl.LazyFrame, observations: pl.DataFrame
    ) -> pl.LazyFrame:
        return lazy_responses

    def customize_responses(
        self, pivoted_responses: pl.DataFrame, observations: pl.DataFrame
    ) -> pl.DataFrame:
        return pivoted_responses

    def _extract_lazy_responses(
        self, realizations: tuple[int, ...], observations: pl.DataFrame
    ) -> pl.LazyFrame:
        responses = self._load_lazy_responses(realizations)
        responses = self.customize_lazy_responses(responses, observations)
        if self.match_config.filter_out_unmatchable_responses:
            responses = self._filter_out_unmatchable_responses(
                responses, observations, self.match_keys
            )
        return responses

    def _load_lazy_responses(self, realizations: tuple[int, ...]) -> pl.LazyFrame:
        if self.match_config.response_key_to_filter_response_on is not None:
            response_id = self.match_config.response_key_to_filter_response_on
        else:
            response_id = self.response_cls.type
        raw_responses = self.ensemble._load_responses_lazy(response_id, realizations)

        if self.match_config.convert_response_key_to_categorical:
            raw_responses = raw_responses.with_columns(
                [pl.col("response_key").cast(pl.Categorical)]
            )
        return raw_responses

    @staticmethod
    def _filter_out_unmatchable_responses(
        responses: pl.LazyFrame, observations: pl.DataFrame, match_keys: list[str]
    ) -> pl.LazyFrame:
        observed_cols = {k: observations[k].unique() for k in match_keys}

        # Filter out responses without observations
        for col, observed_values in observed_cols.items():
            if col not in {"time", "east", "north"}:
                responses = responses.filter(
                    pl.col(col).is_in(observed_values.implode(), nulls_equal=True)
                )
        return responses

    def _collect_responses(
        self, lazy_responses: pl.LazyFrame, observations: pl.DataFrame
    ) -> pl.DataFrame:
        responses = lazy_responses.collect(engine="streaming")
        responses = self.customize_responses(responses, observations)
        responses = self._pivot_responses(responses)
        return responses  # ruff: ignore[unnecessary-assign]

    def _pivot_responses(self, responses: pl.DataFrame) -> pl.DataFrame:
        if self.match_config.pivot_responses_on is None:
            return responses

        on = self.match_config.pivot_responses_on
        if responses[on].dtype == pl.String:
            responses = responses.with_columns(
                pl.col(on).str.to_lowercase(),
            )
        index = list(self.match_keys)
        # realization information should not be lost
        if on != "realization":
            index.append("realization")

        return responses.pivot(
            on=on,
            # match keys should be subset of index to assure row uniqueness
            index=index,
            values="values",
            # backup action to assure one row per observation.
            # shouldn't be used if all values in the index are unique (they should be)
            aggregate_function="mean",
        )


class GenDataMatchStrategy(ReservoirDataMatchStrategy[GenDataConfig]):
    pass


class SummaryMatchStrategy(ReservoirDataMatchStrategy[SummaryConfig]):
    @override
    def match(
        self,
        observations: pl.DataFrame,
        responses: pl.DataFrame,
    ) -> pl.DataFrame:
        by_cols = [*[k for k in self.match_keys if k != "time"]]

        return observations.sort(
            by=[*by_cols, "time"]
        ).join_asof(
            responses.sort(by=[*by_cols, "time"]),
            by=by_cols,
            on="time",
            check_sortedness=False,  # Ref: https://github.com/pola-rs/polars/issues/21693
            strategy="nearest",
            tolerance="1s",
        )


class BreakthroughMatchStrategy(ReservoirDataMatchStrategy[BreakthroughConfig]):
    def __init__(
        self,
        ensemble: LocalEnsemble,
        response_cls: BreakthroughConfig,
        match_config: MatchConfig,
    ) -> None:
        self.ensemble_end_date = None
        if match_config.breakthrough.fill_in_unreached_threshold:
            self.ensemble_end_date = self._compute_ensemble_end_date(ensemble)
        super().__init__(ensemble, response_cls, match_config=match_config)

    @staticmethod
    def _compute_ensemble_end_date(ensemble: LocalEnsemble) -> datetime | None:
        """Returns the last simulated time across all realizations with
        responses.
        All summary keys share the same time axis within a given
        realization, guaranteed by the summary file format.
        """
        if "summary" not in ensemble.experiment.response_configuration:
            return None

        available_realizations = ensemble.get_realization_list_with_responses()
        if not available_realizations:
            return None

        times = ensemble.load_responses("summary", tuple(available_realizations))[
            "time"
        ]
        if len(times) == 0:
            return None
        return times.max()  # type: ignore

    @override
    def customize_lazy_responses(
        self, lazy_responses: pl.LazyFrame, observations: pl.DataFrame
    ) -> pl.LazyFrame:
        if self.match_config.breakthrough.fill_in_unreached_threshold:
            if self.match_config.convert_response_key_to_categorical:
                lazy_responses = lazy_responses.with_columns(
                    pl.col("response_key").cast(pl.String)
                )
            lazy_responses = self.response_cls.fill_missing_values(
                lazy_responses, self.ensemble_end_date
            )
            if self.match_config.convert_response_key_to_categorical:
                lazy_responses = lazy_responses.with_columns(
                    pl.col("response_key").cast(pl.Categorical)
                )
        return lazy_responses


class RFTMatchStrategy(ReservoirDataMatchStrategy[RFTConfig]):
    @override
    def enrich_observations_with_realization_metadata(
        self, observations: pl.DataFrame, realization: int
    ) -> pl.DataFrame:
        observation_metadata_in_realization = (
            self.ensemble.load_observation_location_metadata(realization)
        )
        observations = RFTConfig.enrich_observations_with_metadata(
            observations, observation_metadata_in_realization
        )

        if self.match_config.qc_observations:
            observations = ensure_qc_error_column(observations)
            observations = qc_rft_observations(observations)

        return observations

    @override
    def customize_lazy_responses(
        self, lazy_responses: pl.LazyFrame, observations: pl.DataFrame
    ) -> pl.LazyFrame:
        if (
            self.match_config.rft.approximate_values
            and self.response_cls.approximate_missing_values
        ):
            lazy_responses = RFTConfig.approximate_missing_rft_responses(
                lazy_responses, observations
            )
        return lazy_responses


class SeismicMatchStrategy(ReservoirDataMatchStrategy[SeismicConfig]):
    @override
    def customize_observations(self, observations: pl.DataFrame) -> pl.DataFrame:
        if self.match_config.qc_observations:
            observations = qc_seismic_observations(
                observations, self.ensemble.experiment.shape_registry
            )
        return observations

    @override
    def customize_responses(
        self, pivoted_responses: pl.DataFrame, observations: pl.DataFrame
    ) -> pl.DataFrame:
        return SeismicData.use_observation_locations_in_respective_responses(
            pivoted_responses, observations
        )


MatchStrategyType = (
    GenDataMatchStrategy
    | SummaryMatchStrategy
    | BreakthroughMatchStrategy
    | RFTMatchStrategy
    | SeismicMatchStrategy
)


def matching_strategy(
    ensemble: LocalEnsemble,
    response_cls: ResponseConfig,
    match_config: MatchConfig,
) -> MatchStrategyType:
    """Get strategy for reservoir data matching for provided ResponseConfig."""
    data_map: dict[str, type] = {
        "gen_data": GenDataMatchStrategy,
        "summary": SummaryMatchStrategy,
        "breakthrough": BreakthroughMatchStrategy,
        "rft": RFTMatchStrategy,
        "seismic": SeismicMatchStrategy,
    }
    strategy_cls = data_map[response_cls.type]
    return strategy_cls(ensemble, response_cls, match_config)
