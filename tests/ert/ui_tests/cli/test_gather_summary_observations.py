import subprocess
from argparse import Namespace
from pathlib import Path
from subprocess import CalledProcessError
from textwrap import dedent

import polars as pl
import pytest

from ert.gather_summary_observations import convert_summary_observations, main


@pytest.fixture(name="use_feature_flag")
def use_feature_flag(monkeypatch):
    monkeypatch.setenv("ERT_FEATURE_GATHER_OBS", "1")


@pytest.mark.usefixtures("copy_snake_oil_case_storage", "use_tmpdir")
def test_that_cli_command_without_feature_flag_raises_called_process_error() -> None:
    config_path = "test_data/snake_oil.ert"
    storage_path = "test_data/storage/"
    experiment_path = "snake_oil/ensemble/experiments/"
    experiment = next(iter(Path(storage_path + experiment_path).iterdir()))

    with pytest.raises(CalledProcessError) as e:
        subprocess.run(
            ["ert", "gather_summary_observations", config_path, experiment.name],
            check=True,
        )
    assert e.value.returncode == 2


@pytest.mark.usefixtures(
    "copy_snake_oil_case_storage", "use_tmpdir", "use_feature_flag"
)
@pytest.mark.skip_mac_ci  # Ert api is too slow to start for mac tests
def test_that_gather_summary_observations_outputs_csv_containing_observation_data():
    config_path = "test_data/snake_oil.ert"
    storage_path = "test_data/storage/"
    experiment_path = "snake_oil/ensemble/experiments/"
    experiment = next(iter(Path(storage_path + experiment_path).iterdir()))

    args = Namespace(
        config=config_path,
        experiment=experiment.name,
        output_csv_file="summary_observations.csv",
    )
    main(args)
    assert Path("summary_observations.csv").is_file()
    csv_content = Path("summary_observations.csv").read_text(encoding="utf-8")
    expected_csv_content = [
        "keyword, well, value, error, date",
        "WOPR, OP1, 0.7, 0.07, 2010-12-26",
        "WOPR, OP1, 0.2, 0.035, 2013-12-10",
        "WOPR, OP1, 0.1, 0.05, 2010-03-31",
        "WOPR, OP1, 0.5, 0.05, 2011-12-21",
        "WOPR, OP1, 0.015, 0.01, 2015-03-15",
        "WOPR, OP1, 0.3, 0.075, 2012-12-15",
    ]
    assert all(line in csv_content for line in expected_csv_content)


@pytest.mark.usefixtures(
    "copy_snake_oil_case_storage", "use_tmpdir", "use_feature_flag"
)
@pytest.mark.skip_mac_ci  # Ert api is too slow to start for mac tests
def test_that_single_experiment_in_storage_is_automatically_selected(capsys):
    config_path = "test_data/snake_oil.ert"
    args = Namespace(
        config=config_path,
        experiment=None,
        output_csv_file="summary_observations.csv",
    )
    main(args)

    storage_path = "test_data/storage/"
    experiment_path = "snake_oil/ensemble/experiments/"
    experiment = next(iter(Path(storage_path + experiment_path).iterdir()))
    assert (
        f"Only one experiment found, picking experiment with id '{experiment.name}'"
        in capsys.readouterr().out
    )
    assert Path("summary_observations.csv").is_file()
    csv_content = Path("summary_observations.csv").read_text(encoding="utf-8")
    expected_csv_content = [
        "keyword, well, value, error, date",
        "WOPR, OP1, 0.7, 0.07, 2010-12-26",
        "WOPR, OP1, 0.2, 0.035, 2013-12-10",
        "WOPR, OP1, 0.1, 0.05, 2010-03-31",
        "WOPR, OP1, 0.5, 0.05, 2011-12-21",
        "WOPR, OP1, 0.015, 0.01, 2015-03-15",
        "WOPR, OP1, 0.3, 0.075, 2012-12-15",
    ]
    assert all(line in csv_content for line in expected_csv_content)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_convert_summary_observations_extracts_localization_information(capsys):
    summary_obs = {
        "WOPR:WELL_WITH_LOCALIZATION": [
            {
                "name": "WOPR_OP1_72",
                "errors": [0.02],
                "values": [0.5],
                "east": [None],
                "north": [None],
                "radius": [None],
                "x_axis": ["2010-01-27T00:00:00"],
            },
            {
                "name": "WOPR_OP1_9",
                "errors": [0.05],
                "values": [0.1],
                "east": [10],
                "north": [20],
                "radius": [2500],
                "x_axis": ["2010-03-31T00:00:00"],
            },
        ],
        "WSPC:WELL_WITHOUT_RADIUS": [
            {
                "name": "WSPC",
                "errors": [0.04],
                "values": [3],
                "east": [40],
                "north": [50],
                "radius": [None],
                "x_axis": ["2011-12-21T00:00:00"],
            },
        ],
        "WSIR:WELL_WITHOUT_LOCALIZATION": [
            {
                "name": "WSIR",
                "errors": [0.04],
                "values": [3],
                "x_axis": ["2011-12-21T00:00:00"],
            },
        ],
    }
    convert_summary_observations(
        summary_observations=summary_obs,
        breakthrough_observations={},
        csv_file_name="foo.csv",
    )
    expected_print_with_localization = dedent("""\
    SUMMARY {
      VALUES = foo.csv;
      WELL WELL_WITHOUT_RADIUS {
        LOCALIZATION {
          EAST=40;
          NORTH=50;
        };
      };
      WELL WELL_WITH_LOCALIZATION {
        LOCALIZATION {
          EAST=10;
          NORTH=20;
          RADIUS=2500;
        };
      };
    };""")
    assert expected_print_with_localization in capsys.readouterr().out


@pytest.mark.usefixtures(
    "copy_snake_oil_case_storage", "use_tmpdir", "use_feature_flag"
)
def test_that_gather_summary_obs_can_gather_well_localization_from_breakthrough(capsys):
    config_path = "test_data/snake_oil.ert"
    storage_path = "test_data/storage/"
    experiment_path = "snake_oil/ensemble/experiments/"
    experiment = next(iter(Path(storage_path + experiment_path).iterdir()))
    obs_folder = Path(experiment / "observations")
    breakthrough_obs = pl.DataFrame(
        {
            "response_key": ["BREAKTHROUGH:WWCT:OP1"],
            "observation_key": ["BRT_OP1"],
            "time": ["2012-10-10"],
            "threshold": pl.Series([0.2], dtype=pl.Float64),
            "observations": pl.Series([0.0], dtype=pl.Float32),
            "std": pl.Series([5.0], dtype=pl.Float32),
            "east": pl.Series([100.0], dtype=pl.Float32),
            "north": pl.Series([200.0], dtype=pl.Float32),
            "radius": pl.Series([2500.0], dtype=pl.Float32),
        }
    )
    breakthrough_obs.write_parquet(Path(obs_folder / "breakthrough"))
    args = Namespace(
        config=config_path,
        experiment=experiment.name,
        output_csv_file="summary_observations.csv",
    )
    main(args)
    expected_stdout = dedent("""\
    SUMMARY {
      VALUES = summary_observations.csv;
      WELL OP1 {
        LOCALIZATION {
          EAST=100.0;
          NORTH=200.0;
          RADIUS=2500.0;
        };
        BREAKTHROUGH {
          THRESHOLD=0.2;
          DATE=2012-10-10T00:00:00;
          ERROR=5.0;
          KEY=WWCT;
        };
      };
    };""")
    assert expected_stdout in capsys.readouterr().out
