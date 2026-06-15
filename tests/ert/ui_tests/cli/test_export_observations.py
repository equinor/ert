import shutil
import subprocess
from argparse import Namespace
from pathlib import Path
from subprocess import CalledProcessError

import pytest

from ert.config import ErtConfig
from ert.export_observations.export_observations import main


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
            ["ert", "export_observations", config_path, experiment.name],
            check=True,
        )
    assert e.value.returncode == 2


@pytest.mark.usefixtures(
    "copy_snake_oil_case_storage", "use_tmpdir", "use_feature_flag"
)
@pytest.mark.skip_mac_ci  # Ert api is too slow to start for mac tests
def test_that_happy_path_on_snake_oil_produces_csv_and_stdout(capsys):
    """This tests that the produced stdout and csv file from the
    gather_summary_observations command is what we expect.
    Finally, the test also creates a ErtConfig from the initial observation config
    and compares it to the new one when replacing the summary observations with the
    stdout and moving the csv file into the observations folder.
    """
    config_path = "test_data/snake_oil.ert"
    storage_path = "test_data/storage/"
    experiment_path = "snake_oil/ensemble/experiments/"
    observation_path = "test_data/observations/observations.txt"
    experiment = next(iter(Path(storage_path + experiment_path).iterdir()))

    args = Namespace(
        config=config_path,
        experiment_id=experiment.name,
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

    expected_stdout = "SUMMARY {\n  VALUES = summary_observations.csv;\n};\n"
    stdout = capsys.readouterr().out
    assert expected_stdout in stdout

    old_obs_config = Path(observation_path).read_text(encoding="utf-8")
    old_obs_config_lines = old_obs_config.split("\n")
    # Assumes General Observation at bottom of file
    summary_stop_line = next(
        i
        for i, line in enumerate(old_obs_config_lines)
        if "GENERAL_OBSERVATION" in line
    )
    non_bulk_observations = old_obs_config_lines[summary_stop_line:]

    stdout_lines = stdout.split("\n")
    bulk_start_line = next(
        i for i, line in enumerate(stdout_lines) if "SUMMARY {" in line
    )
    extracted_bulk_lines = stdout_lines[bulk_start_line:]

    new_obs_content = "\n".join([*extracted_bulk_lines, *non_bulk_observations])

    old_ert_config = ErtConfig.from_file("test_data/snake_oil.ert")

    Path(observation_path).write_text(new_obs_content, encoding="utf-8")
    shutil.move(
        "summary_observations.csv", "test_data/observations/summary_observations.csv"
    )
    new_ert_config = ErtConfig.from_file("test_data/snake_oil.ert")

    assert len(new_ert_config.observation_declarations) == len(
        old_ert_config.observation_declarations
    )
    # Loop through observations and assert that they are the same except name,
    # which also indirectly asserts the ordering is the same.
    for i in range(len(new_ert_config.observation_declarations)):
        old_obs = old_ert_config.observation_declarations[i].__dict__
        new_obs = new_ert_config.observation_declarations[i].__dict__
        # Bulk summary config must utilize default naming for observations,
        # so these will differ between the observation declarations.
        old_obs.pop("name")
        new_obs.pop("name")
        assert old_obs == new_obs
