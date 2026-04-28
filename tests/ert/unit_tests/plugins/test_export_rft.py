from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

from ert.config import RFTConfig
from ert.config._observations import RFTObservation
from ert.plugins import ErtPluginManager
from ert.plugins.hook_implementations.workflows.export_rft import (
    ExportRFTJob,
)
from ert.storage import open_storage


@contextmanager
def _create_rft_ensemble(ensemble_size):
    rft_config = RFTConfig(input_files=["DUMMY"])
    observations = [
        RFTObservation(
            name="obs1",
            well="WELL1",
            date="2020-01-01",
            property="PRESSURE",
            value=150.0,
            error=5.0,
            north=200.0,
            east=100.0,
            tvd=25.0,
            md=50.0,
            zone=None,
        )
    ]
    with open_storage("storage", mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [rft_config.model_dump(mode="json")],
                "observations": [o.model_dump(mode="json") for o in observations],
            }
        )
        yield storage.create_ensemble(
            experiment.id, ensemble_size=ensemble_size, name="test"
        )


def _create_rft_response_df(
    well: str = "WELL1",
    date: str = "2020-01-01",
    prop: str = "PRESSURE",
    depth: float = 8000.0,
    value: float = 148.0,
    i: int = 1,
    j: int = 2,
    k: int = 3,
) -> pl.DataFrame:
    time = datetime.strptime(date, "%Y-%m-%d").date()  # noqa: DTZ007
    df = pl.DataFrame(
        {
            "response_key": [f"{well}:{date}:{prop}"],
            "well": [well],
            "date": [date],
            "property": [prop],
            "time": [time],
            "depth": pl.Series([depth], dtype=pl.Float32),
            "values": pl.Series([value], dtype=pl.Float32),
            "well_connection_cell": pl.Series([(i, j, k)], dtype=pl.Array(pl.Int64, 3)),
        }
    )
    return RFTConfig._assert_schema(df, RFTConfig.response_schema())


def _create_rft_location_metadata_df(
    *,
    east: float = 100.0,
    north: float = 200.0,
    tvd: float = 25.0,
    zone: str | None = None,
    i: int = 1,
    j: int = 2,
    k: int = 3,
) -> pl.DataFrame:
    zones = [zone] if zone is not None else []
    df = pl.DataFrame(
        {
            "east": pl.Series([east], dtype=pl.Float32),
            "north": pl.Series([north], dtype=pl.Float32),
            "tvd": pl.Series([tvd], dtype=pl.Float32),
            "actual_zones": pl.Series([zones], dtype=pl.List(pl.String)),
            "well_connection_cell": pl.Series([(i, j, k)], dtype=pl.Array(pl.Int64, 3)),
        }
    )
    return RFTConfig._assert_schema(df, RFTConfig.location_metadata_schema())


def test_that_export_rft_job_is_registered_in_plugin_manager():
    pm = ErtPluginManager()
    assert "EXPORT_RFT" in pm.get_ertscript_workflows().get_workflows()


def _mock_runpath(runpaths):
    run_paths = MagicMock()
    run_paths.get_paths.return_value = runpaths
    return run_paths


@pytest.mark.usefixtures("use_tmpdir")
def test_that_export_rft_writes_csv_files_to_runpaths():

    runpath_values = [
        (Path("real0"), _create_rft_response_df()),
        (Path("real1"), _create_rft_response_df(value=152.0)),
    ]

    for rp, _ in runpath_values:
        rp.mkdir()

    with _create_rft_ensemble(ensemble_size=2) as ensemble:
        for i, (_, response) in enumerate(runpath_values):
            ensemble.save_response("rft", response, i)
            ensemble.save_observation_location_metadata(
                _create_rft_location_metadata_df(), i
            )

        ExportRFTJob().run(
            _mock_runpath([str(rp) for rp, _ in runpath_values]), ensemble, []
        )

        for runpath, response in runpath_values:
            output_file = runpath / "share/results/tables/rft_ert.csv"

            assert output_file.exists()

            output = pl.read_csv(output_file)

            assert "realization" not in output.columns

            assert output["pressure"][0] == response["values"][0]


@pytest.mark.usefixtures("use_tmpdir")
def test_that_export_rft_uses_custom_filename():
    responses_real0 = _create_rft_response_df()

    runpath0 = Path("real0")
    runpath0.mkdir()

    with _create_rft_ensemble(ensemble_size=1) as ensemble:
        realization = 0
        ensemble.save_response("rft", responses_real0, realization)
        ensemble.save_observation_location_metadata(
            _create_rft_location_metadata_df(),
            realization,
        )

        ExportRFTJob().run(_mock_runpath([str(runpath0)]), ensemble, ["custom_rft.csv"])

        output_file = runpath0 / "custom_rft.csv"
        assert output_file.exists()

        assert pl.read_csv(output_file)["pressure"][0] == responses_real0["values"][0]
