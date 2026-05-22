import subprocess
from pathlib import Path

import pytest


@pytest.mark.usefixtures("copy_snake_oil_case_storage", "use_tmpdir")
def test_that_cli_command_outputs_csv_containing_observation_data():
    config_path = "test_data/snake_oil.ert"
    storage_path = "test_data/storage/"
    experiment_path = "snake_oil/ensemble/experiments/"
    experiment = next(iter(Path(storage_path + experiment_path).iterdir()))
    subprocess.run(
        ["ert", "gather_summary_observations", config_path, experiment.name], check=True
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
