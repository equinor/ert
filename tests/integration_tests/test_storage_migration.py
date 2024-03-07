import shutil
from pathlib import Path

import pytest

from ert.config import ErtConfig
from ert.storage import open_storage
from ert.storage.local_storage import local_storage_set_ert_config


@pytest.mark.xfail(reason="github.com/equinor/ert/issues/7388")
def test_that_storage_matches(tmp_path, source_root, snapshot, monkeypatch):
    shutil.copytree(
        Path(source_root) / "test-data/block_storage/all_data_types/",
        tmp_path / "all_data_types",
    )
    monkeypatch.chdir(tmp_path / "all_data_types")
    ert_config = ErtConfig.from_file("config.ert")
    local_storage_set_ert_config(ert_config)
    for ert_version in [
        "8.4.9",
        "8.4.8",
        "8.4.7",
        "8.4.6",
        "8.4.5",
        "8.4.4",
        "8.4.3",
        "8.4.2",
        "8.4.1",
        "8.4.0",
        "8.3.1",
        "8.3.0",
        "8.2.1",
        "8.2.0",
        "8.1.1",
        "8.1.0",
        "8.0.13",
        "8.0.12",
        "8.0.11",
        "8.0.10",
        "8.0.9",
        "8.0.8",
        "8.0.7",
        "8.0.6",
    ]:
        print(ert_version)
        with open_storage(f"storage-{ert_version}", "w") as storage:
            experiments = list(storage.experiments)
            assert len(experiments) == 1
            experiment = experiments[0]
            ensembles = list(experiment.ensembles)
            assert len(ensembles) == 1

            # We need to normalize some irrelevant details:
            experiment.response_configuration["summary"].refcase = {}
            experiment.response_configuration["summary"].keys = sorted(
                set(experiment.response_configuration["summary"].keys)
            )
            experiment.parameter_configuration["PORO"].mask_file = ""

            snapshot.assert_match(
                str(experiment.parameter_configuration) + "\n",
                "parameters",
            )
            snapshot.assert_match(
                str(experiment.response_configuration) + "\n", "responses"
            )
