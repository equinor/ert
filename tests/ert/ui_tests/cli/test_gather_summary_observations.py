import subprocess
from argparse import Namespace
from pathlib import Path
from subprocess import CalledProcessError

import pytest

from ert.gather_summary_observations import main


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
def test_that_happy_path_on_snake_oil_produces_csv_and_stdout(capsys):
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

    expected_stdout = "SUMMARY {\n  VALUES = summary_observations.csv;\n};\n"
    assert expected_stdout in capsys.readouterr().out
