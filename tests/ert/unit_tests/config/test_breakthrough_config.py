from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl
import pytest

from ert.config import BreakthroughConfig, SummaryConfig
from ert.storage import Ensemble
from ert.storage.local_storage import open_storage


@dataclass
class BreakthroughData:
    response_key: str
    time: datetime
    breakthrough_config: BreakthroughConfig

    @staticmethod
    def monthly_times() -> list[datetime]:
        return [
            datetime(2000, month, 1, 1, 0)  # ruff: ignore[call-datetime-without-tzinfo]
            for month in range(1, 6)
        ]

    def ensemble_end_date(self) -> datetime:
        return max(self.monthly_times())

    def values_never_reaching_threshold(self, realization: int = 0) -> pl.DataFrame:
        times = self.monthly_times()
        return pl.DataFrame(
            {
                "realization": [realization] * len(times),
                "response_key": [self.response_key] * len(times),
                "time": times,
                "values": [n / 100 for n in range(len(times))],
            }
        )

    def response_df(
        self,
        response_keys: list[str] | None = None,
        thresholds: list[float] | None = None,
        times: list[datetime | None] | None = None,
        values: list[float | None] | None = None,
    ) -> pl.LazyFrame:
        response_keys = (
            [f"BREAKTHROUGH:{self.response_key}"]
            if response_keys is None
            else response_keys
        )
        count = len(response_keys)
        return pl.DataFrame(
            {
                "response_key": response_keys,
                "threshold": [0.2] * count if thresholds is None else thresholds,
                "time": [None] * count if times is None else times,
                "values": [None] * count if values is None else values,
            },
            schema=self.breakthrough_config.response_schema(),
        ).lazy()


@dataclass
class BreakthroughFixture(BreakthroughData):
    ensemble: Ensemble


def _breakthrough_config(
    keys: list[str] | None = None,
    summary_keys: list[str] | None = None,
    thresholds: list[float] | None = None,
    observed_dates: list[datetime] | None = None,
) -> BreakthroughConfig:
    return BreakthroughConfig(
        keys=["BREAKTHROUGH:OP1"] if keys is None else keys,
        summary_keys=["WWCT:OP1"] if summary_keys is None else summary_keys,
        thresholds=[0.2] if thresholds is None else thresholds,
        observed_dates=[
            datetime(2000, 3, 2, 13, 0)  # ruff: ignore[call-datetime-without-tzinfo]
        ]
        if observed_dates is None
        else observed_dates,
    )


def _breakthrough_data(
    response_key="WWCT:OP1",
    time=datetime(2000, 3, 2, 13, 0),  # ruff: ignore[call-datetime-without-tzinfo]
) -> BreakthroughData:
    breakthrough_config = BreakthroughConfig(
        keys=[f"BREAKTHROUGH:{response_key}"],
        summary_keys=[response_key],
        thresholds=[0.2],
        observed_dates=[time],
    )

    return BreakthroughData(
        response_key=response_key, time=time, breakthrough_config=breakthrough_config
    )


@contextmanager
def _breakthrough_setup(tmp_path, breakthrough_data=None):
    breakthrough_data = (
        _breakthrough_data() if breakthrough_data is None else breakthrough_data
    )

    with open_storage(tmp_path, mode="w") as storage:
        summary_config = SummaryConfig(
            keys=[breakthrough_data.response_key],
            input_files=["not_relevant"],
        )

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [
                    summary_config.model_dump(mode="json"),
                    breakthrough_data.breakthrough_config.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=2, iteration=0, name="prior"
        )
        yield BreakthroughFixture(
            response_key=breakthrough_data.response_key,
            time=breakthrough_data.time,
            breakthrough_config=breakthrough_data.breakthrough_config,
            ensemble=ensemble,
        )


def test_that_derive_from_storage_frames_are_stackable_regardless_of_breakthrough_time(
    tmp_path,
):
    with _breakthrough_setup(tmp_path) as breakthrough_setup:
        response_key = breakthrough_setup.response_key
        breakthrough_config = breakthrough_setup.breakthrough_config
        ensemble = breakthrough_setup.ensemble

        def create_summary_response_dataframe(
            response_key: str, realization: int, value_modifier
        ) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "realization": [realization] * 5,
                    "response_key": [response_key] * 5,
                    "time": [datetime(2000, month, 1, 1, 0) for month in range(1, 6)],  # ruff: ignore[call-datetime-without-tzinfo]
                    "values": [n / value_modifier for n in range(5)],
                }
            )

        value_over_threshold = create_summary_response_dataframe(response_key, 0, 10)
        value_under_threshold = create_summary_response_dataframe(response_key, 1, 100)

        ensemble.save_response("summary", value_over_threshold, 0)
        ensemble.save_response("summary", value_under_threshold, 1)

        breakthrough_response0 = breakthrough_config.derive_from_storage(0, 0, ensemble)
        breakthrough_response1 = breakthrough_config.derive_from_storage(0, 1, ensemble)

        ensemble.save_response("breakthrough", breakthrough_response0, 0)
        ensemble.save_response("breakthrough", breakthrough_response1, 1)

        responses = ensemble.load_responses("breakthrough", (0, 1))
        assert len(responses) == 2


def test_that_derive_from_storage_stores_null_when_threshold_is_not_reached(
    tmp_path,
):
    with _breakthrough_setup(tmp_path) as breakthrough_setup:
        breakthrough_config = breakthrough_setup.breakthrough_config
        ensemble = breakthrough_setup.ensemble

        ensemble.save_response(
            "summary", breakthrough_setup.values_never_reaching_threshold(), 0
        )

        breakthrough_response = breakthrough_config.derive_from_storage(0, 0, ensemble)

        assert breakthrough_response["time"].item() is None
        assert breakthrough_response["values"].item() is None


def test_that_loading_breakthrough_responses_fills_in_with_day_after_ensemble_end_date(
    tmp_path,
):
    with _breakthrough_setup(tmp_path) as breakthrough_setup:
        ensemble = breakthrough_setup.ensemble

        ensemble.save_response(
            "summary", breakthrough_setup.values_never_reaching_threshold(), 0
        )

        breakthrough_response = (
            breakthrough_setup.breakthrough_config.derive_from_storage(0, 0, ensemble)
        )
        ensemble.save_response("breakthrough", breakthrough_response, 0)

        loaded = ensemble.load_responses("breakthrough", (0,))

        assert loaded["time"].item() is not None
        expected_offset_days = (
            breakthrough_setup.ensemble_end_date()
            + timedelta(days=1)
            - breakthrough_setup.time
        ).total_seconds() / (60 * 60 * 24)
        assert loaded["values"].item() == pytest.approx(expected_offset_days)


def test_that_fill_missing_values_returns_unchanged_when_there_are_no_summary_keys():
    breakthrough_config = _breakthrough_config(
        keys=[], summary_keys=[], thresholds=[], observed_dates=[]
    )

    any_response_df = _breakthrough_data().response_df()

    result = breakthrough_config.fill_missing_values(
        any_response_df, ensemble_end_date=_breakthrough_data().ensemble_end_date()
    )
    assert result.collect().equals(any_response_df.collect())


def test_that_fill_missing_values_returns_unchanged_when_no_values_are_missing():
    breakthrough_data = _breakthrough_data()
    response_df = breakthrough_data.response_df(
        times=[breakthrough_data.time], values=[1.5]
    )

    result = _breakthrough_config().fill_missing_values(
        response_df, ensemble_end_date=_breakthrough_data().ensemble_end_date()
    )
    assert result.collect().equals(response_df.collect())


def test_that_fill_missing_values_returns_unchanged_when_ensemble_end_date_is_none():
    any_response_df = _breakthrough_data().response_df()

    result = _breakthrough_config().fill_missing_values(
        any_response_df, ensemble_end_date=None
    )
    assert result.collect().equals(any_response_df.collect())


def test_that_fill_missing_values_leaves_already_filled_rows_untouched():
    filled_key, unfilled_key = "BREAKTHROUGH:OP1", "BREAKTHROUGH:OP2"
    obs_date_filled = datetime(2000, 3, 2, 13, 0)  # ruff: ignore[call-datetime-without-tzinfo]
    obs_date_unfilled = datetime(2000, 4, 2, 13, 0)  # ruff: ignore[call-datetime-without-tzinfo]
    breakthrough_time = datetime(2000, 4, 15, 1, 0)  # ruff: ignore[call-datetime-without-tzinfo]
    ensemble_end_date = datetime(2000, 5, 1, 1, 0)  # ruff: ignore[call-datetime-without-tzinfo]

    breakthrough_config = _breakthrough_config(
        keys=[filled_key, unfilled_key],
        summary_keys=["WWCT:OP1", "WWCT:OP2"],
        thresholds=[0.2, 0.2],
        observed_dates=[obs_date_filled, obs_date_unfilled],
    )
    response_df = _breakthrough_data().response_df(
        response_keys=[filled_key, unfilled_key],
        times=[breakthrough_time, None],
        values=[1.5, None],
    )

    result = breakthrough_config.fill_missing_values(
        response_df, ensemble_end_date=ensemble_end_date
    ).collect()

    already_filled_row = result.filter(pl.col("response_key") == filled_key)
    assert already_filled_row["time"].item() == breakthrough_time
    assert already_filled_row["values"].item() == pytest.approx(1.5)

    newly_filled_row = result.filter(pl.col("response_key") == unfilled_key)
    assert newly_filled_row["time"].item() is not None
    expected_offset_days = (
        ensemble_end_date + timedelta(days=1) - obs_date_unfilled
    ).total_seconds() / (60 * 60 * 24)
    assert newly_filled_row["values"].item() == pytest.approx(expected_offset_days)


def test_that_fill_missing_values_matches_on_threshold_when_summary_key_is_shared():
    shared_response_key = "BREAKTHROUGH:WWCT:OP1"
    summary_key = "WWCT:OP1"
    obs_date_low_threshold = datetime(2000, 3, 2, 13, 0)  # ruff: ignore[call-datetime-without-tzinfo]
    obs_date_high_threshold = datetime(2000, 4, 2, 13, 0)  # ruff: ignore[call-datetime-without-tzinfo]
    ensemble_end_date = datetime(2000, 5, 1, 1, 0)  # ruff: ignore[call-datetime-without-tzinfo]

    breakthrough_config = _breakthrough_config(
        keys=[shared_response_key, shared_response_key],
        summary_keys=[summary_key, summary_key],
        thresholds=[0.2, 0.5],
        observed_dates=[obs_date_low_threshold, obs_date_high_threshold],
    )
    response_df = _breakthrough_data().response_df(
        response_keys=[shared_response_key, shared_response_key],
        thresholds=[0.2, 0.5],
    )

    result = breakthrough_config.fill_missing_values(
        response_df, ensemble_end_date=ensemble_end_date
    ).collect()

    assert len(result) == 2

    def expected_offset_days(obs_date: datetime) -> float:
        return (ensemble_end_date + timedelta(days=1) - obs_date).total_seconds() / (
            60 * 60 * 24
        )

    low_threshold_row = result.filter((pl.col("threshold") - 0.2).abs() < 1e-9)
    assert low_threshold_row["values"].item() == pytest.approx(
        expected_offset_days(obs_date_low_threshold)
    )

    high_threshold_row = result.filter((pl.col("threshold") - 0.5).abs() < 1e-9)
    assert high_threshold_row["values"].item() == pytest.approx(
        expected_offset_days(obs_date_high_threshold)
    )
