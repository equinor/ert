import datetime

import pandas as pd
import polars as pl
import pytest

from ert.config import GenDataConfig, SummaryConfig
from ert.dark_storage.common import data_for_key
from ert.storage import open_storage
from tests.ert.unit_tests.config.summary_generator import (
    Date,
    Simulator,
    Smspec,
    SmspecIntehead,
    SummaryMiniStep,
    SummaryStep,
    UnitSystem,
    Unsmry,
)


def test_data_for_key_gives_mean_for_duplicate_values(tmp_path):
    value1 = 1.1
    value2 = 1.0e19
    with open_storage(tmp_path / "storage", mode="w") as storage:
        summary_config = SummaryConfig(name="summary", input_files=["CASE"], keys=["*"])
        experiment = storage.create_experiment(
            observations={},
            parameters=[],
            responses=[summary_config],
        )
        ensemble = experiment.create_ensemble(name="ensemble", ensemble_size=1)
        unsmry = Unsmry(
            steps=[
                SummaryStep(
                    seqnum=0,
                    ministeps=[
                        SummaryMiniStep(mini_step=0, params=[0.0, 5.629901e16]),
                        SummaryMiniStep(mini_step=1, params=[365.0, value1]),
                    ],
                ),
                SummaryStep(
                    seqnum=1,
                    ministeps=[SummaryMiniStep(mini_step=2, params=[365.0, value2])],
                ),
            ]
        )
        smspec = Smspec(
            nx=4,
            ny=4,
            nz=10,
            restarted_from_step=0,
            num_keywords=2,
            restart="        ",
            keywords=["TIME    ", "NRPPR"],
            well_names=[":+:+:+:+", "WELLNAME"],
            region_numbers=[-32676, 0],
            units=["HOURS   ", "SM3"],
            start_date=Date(
                day=1, month=1, year=2014, hour=0, minutes=0, micro_seconds=0
            ),
            intehead=SmspecIntehead(
                unit=UnitSystem.METRIC,
                simulator=Simulator.ECLIPSE_100,
            ),
        )
        smspec.to_file(tmp_path / "CASE.SMSPEC")
        unsmry.to_file(tmp_path / "CASE.UNSMRY")
        ds = summary_config.read_from_file(tmp_path, 0, 0)
        ensemble.save_response(summary_config.response_type, ds, 0)
        df = data_for_key(ensemble, "NRPPR:WELLNAME")
        assert list(df.columns) == [pd.Timestamp("2014-01-16 05:00:00")]
        assert df[pd.Timestamp("2014-01-16 05:00:00")][0] == pytest.approx(
            (value1 + value2) / 2
        )


def test_data_for_key_doesnt_mistake_history_for_response(tmp_path):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        summary_config = SummaryConfig(
            name="summary", input_files=["CASE"], keys=["FGPR"]
        )
        experiment = storage.create_experiment(responses=[summary_config])
        ensemble = experiment.create_ensemble(name="ensemble", ensemble_size=1)
        ensemble.save_response(
            "summary",
            pl.DataFrame(
                {
                    "response_key": ["FGPR", "FGPR"],
                    "time": pl.Series(
                        [datetime.datetime(2000, 1, 1), datetime.datetime(2000, 1, 2)],
                        dtype=pl.Datetime("ms"),
                    ),
                    "values": pl.Series([0.0, 1.0], dtype=pl.Float32),
                }
            ),
            0,
        )
        ensemble.refresh_ensemble_state()

        assert data_for_key(ensemble, "FGPRH").empty


def test_data_for_key_returns_empty_gen_data_config(tmp_path):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        gen_data_config = GenDataConfig(keys=["response"])
        experiment = storage.create_experiment(
            observations={},
            parameters=[],
            responses=[gen_data_config],
        )
        ensemble = experiment.create_ensemble(name="ensemble", ensemble_size=1)

        data = data_for_key(ensemble, "response@0")
        assert data.empty

        ensemble.save_response(
            "gen_data",
            pl.DataFrame(
                {
                    "response_key": "response",
                    "report_step": pl.Series([0], dtype=pl.UInt16),
                    "index": pl.Series([0], dtype=pl.UInt16),
                    "values": pl.Series([0.0], dtype=pl.Float32),
                }
            ),
            0,
        )
        ensemble.refresh_ensemble_state()
        data = data_for_key(ensemble, "response@0")
        assert not data.empty
