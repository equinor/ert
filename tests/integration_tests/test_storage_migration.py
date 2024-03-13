import shutil

import numpy as np
import pytest

from ert.config import ErtConfig
from ert.storage import open_storage
from ert.storage.local_storage import local_storage_set_ert_config


@pytest.fixture()
def copy_shared(tmp_path, block_storage_path):
    for input_dir in ["data", "refcase"]:
        shutil.copytree(
            block_storage_path / "all_data_types" / f"{input_dir}",
            tmp_path / "all_data_types" / f"{input_dir}",
        )
    for file in ["config.ert", "observations.txt", "params.txt", "template.txt"]:
        shutil.copy(
            block_storage_path / f"all_data_types/{file}",
            tmp_path / "all_data_types" / file,
        )


@pytest.mark.usefixtures("copy_shared")
@pytest.mark.parametrize(
    "ert_version",
    [
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
        "8.0.4",
        "8.0.3",
        "8.0.2",
        "8.0.1",
        "8.0.0",
        "7.0.4",
        "7.0.3",
        "7.0.2",
        "7.0.1",
        "7.0.0",
        "6.0.8",
        "6.0.7",
        "6.0.6",
        "6.0.5",
        "6.0.4",
        "6.0.3",
        "6.0.2",
        "6.0.1",
        "6.0.0",
        # github.com/equinor/ert/issues/7401
        # Cannot load due to missing "_ert_kind"
        # "5.0.12",
        # "5.0.11",
        # "5.0.10",
        # "5.0.9",
        # "5.0.8",
        # "5.0.7",
        # "5.0.6",
        # "5.0.5",
        # "5.0.4",
        # "5.0.2",
        # "5.0.1",
        # "5.0.0",
    ],
)
def test_that_storage_matches(
    tmp_path, block_storage_path, snapshot, monkeypatch, ert_version
):
    shutil.copytree(
        block_storage_path / f"all_data_types/storage-{ert_version}",
        tmp_path / "all_data_types" / f"storage-{ert_version}",
    )
    monkeypatch.chdir(tmp_path / "all_data_types")
    ert_config = ErtConfig.from_file("config.ert")
    local_storage_set_ert_config(ert_config)
    # To make sure all tests run against the same snapshot
    snapshot.snapshot_dir = snapshot.snapshot_dir.parent
    with open_storage(f"storage-{ert_version}", "w") as storage:
        experiments = list(storage.experiments)
        assert len(experiments) == 1
        experiment = experiments[0]
        ensembles = list(experiment.ensembles)
        assert len(ensembles) == 1
        ensemble = ensembles[0]

        # We need to normalize some irrelevant details:
        experiment.response_configuration["summary"].refcase = {}
        experiment.parameter_configuration["PORO"].mask_file = ""

        snapshot.assert_match(
            str(experiment.parameter_configuration) + "\n",
            "parameters",
        )
        snapshot.assert_match(
            str(experiment.response_configuration) + "\n", "responses"
        )

        summary_data = ensemble.load_responses(
            "summary",
            tuple(ensemble.get_realization_list_with_responses("summary")),
        )
        snapshot.assert_match(
            summary_data.to_dataframe().astype(np.float32).transform(np.sort).to_csv(),
            "summary_data",
        )
        snapshot.assert_match_dir(
            {
                key: value.to_dataframe().to_csv()
                for key, value in experiment.observations.items()
            },
            "observations",
        )
