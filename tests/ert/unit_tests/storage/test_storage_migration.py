import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from packaging import version

from ert.config import ErtConfig
from ert.storage import open_storage
from ert.storage.local_storage import (
    _LOCAL_STORAGE_VERSION,
    local_storage_set_ert_config,
)
from tests.ert.ui_tests.cli.run_cli import run_cli


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


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_shared")
@pytest.mark.parametrize(
    "ert_version",
    [
        "10.3.1",
        "10.2.8",
        "10.1.3",
        "10.0.3",
        "9.0.17",
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
        "5.0.12",
        "5.0.11",
        "5.0.10",
        "5.0.9",
        "5.0.8",
        "5.0.7",
        "5.0.6",
        "5.0.5",
        "5.0.4",
        "5.0.2",
        "5.0.1",
        "5.0.0",
    ],
)
def test_that_storage_matches(
    tmp_path,
    block_storage_path,
    snapshot,
    monkeypatch,
    ert_version,
):
    shutil.copytree(
        block_storage_path / f"all_data_types/storage-{ert_version}",
        tmp_path / "all_data_types" / f"storage-{ert_version}",
    )
    monkeypatch.chdir(tmp_path / "all_data_types")
    ert_config = ErtConfig.with_plugins().from_file("config.ert")
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

        response_config = experiment.response_configuration
        response_config["summary"].refcase = {}

        with open(
            experiment._path / experiment._responses_file, "w", encoding="utf-8"
        ) as f:
            json.dump(
                {k: v.to_dict() for k, v in response_config.items()},
                f,
                default=str,
                indent=2,
            )

        # We need to normalize some irrelevant details:
        experiment.parameter_configuration["PORO"].mask_file = ""
        if version.parse(ert_version).major == 5:
            # In this version we were not saving the full parameter
            # configuration, so it had to be recreated by what was
            # in ErtConfig at the time of migration, hence the new
            # path
            experiment.parameter_configuration[
                "BPR"
            ].template_file = experiment.parameter_configuration[
                "BPR"
            ].template_file.replace(
                str(tmp_path), "/home/eivind/Projects/ert/test-data"
            )
        snapshot.assert_match(
            str(dict(sorted(experiment.parameter_configuration.items()))) + "\n",
            "parameters",
        )
        snapshot.assert_match(
            str(
                {
                    k: experiment.response_configuration[k]
                    for k in sorted(experiment.response_configuration.keys())
                }
            )
            + "\n",
            "responses",
        )

        summary_data = ensemble.load_responses(
            "summary",
            tuple(ensemble.get_realization_list_with_responses("summary")),
        )
        snapshot.assert_match(
            summary_data.sort("time", "response_key", "realization")
            .to_pandas()
            .set_index(["time", "response_key", "realization"])
            .transform(np.sort)
            .to_csv(),
            "summary_data",
        )
        snapshot.assert_match_dir(
            {
                key: value.to_pandas().to_csv()
                for key, value in experiment.observations.items()
            },
            "observations",
        )

        assert ensemble.get_summary_keyset() == ["FOPR"]


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_shared")
@pytest.mark.parametrize(
    "ert_version",
    [
        "10.3.1",
        "10.2.8",
        "10.1.3",
        "10.0.3",
        "9.0.17",
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
        "5.0.12",
        "5.0.11",
        "5.0.10",
        "5.0.9",
        "5.0.8",
        "5.0.7",
        "5.0.6",
        "5.0.5",
        "5.0.4",
        "5.0.2",
        "5.0.1",
        "5.0.0",
    ],
)
def test_that_storage_works_with_missing_parameters_and_responses(
    tmp_path,
    block_storage_path,
    snapshot,
    monkeypatch,
    ert_version,
):
    storage_path = tmp_path / "all_data_types" / f"storage-{ert_version}"
    shutil.copytree(
        block_storage_path / f"all_data_types/storage-{ert_version}",
        storage_path,
    )
    [ensemble_id] = os.listdir(storage_path / "ensembles")

    ensemble_path = storage_path / "ensembles" / ensemble_id

    # Remove all realization-*/TOP.nc, and only some realization-*/BPC.nc
    for i, real_dir in enumerate(
        (storage_path / "ensembles" / ensemble_id).glob("realization-*")
    ):
        os.remove(real_dir / "TOP.nc")
        if i % 2 == 0:
            os.remove(real_dir / "BPR.nc")

        os.remove(real_dir / "GEN.nc")

    monkeypatch.chdir(tmp_path / "all_data_types")
    ert_config = ErtConfig.with_plugins().from_file("config.ert")
    local_storage_set_ert_config(ert_config)
    # To make sure all tests run against the same snapshot
    snapshot.snapshot_dir = snapshot.snapshot_dir.parent
    with open_storage(f"storage-{ert_version}", "w") as storage:
        experiments = list(storage.experiments)
        assert len(experiments) == 1
        experiment = experiments[0]
        ensembles = list(experiment.ensembles)
        assert len(ensembles) == 1

        ens_dir_contents = set(os.listdir(ensemble_path))
        assert {
            "index.json",
        }.issubset(ens_dir_contents)

        assert "TOP.nc" not in ens_dir_contents

        with pytest.raises(KeyError):
            ensembles[0].load_responses("GEN", (0,))


def test_that_migrate_blockfs_creates_backup_folder(tmp_path, caplog):
    with open(tmp_path / "config.ert", mode="w", encoding="utf-8") as f:
        f.writelines(["NUM_REALIZATIONS 1\n", "ENSPATH", str(tmp_path / "storage")])

    os.makedirs(tmp_path / "storage")
    with open(tmp_path / "storage" / "index.json", "w+", encoding="utf-8") as f:
        f.write("""{"version": 0}""")

    os.makedirs(tmp_path / "storage" / "experiments")
    os.makedirs(tmp_path / "storage" / "ensembles")

    Path(tmp_path / "storage" / "experiments" / "exp_dummy.txt").write_text(
        "", encoding="utf-8"
    )
    Path(tmp_path / "storage" / "ensembles" / "ens_dummy.txt").write_text(
        "", encoding="utf-8"
    )

    with caplog.at_level(level=logging.INFO):
        run_cli("test_run", str(tmp_path / "config.ert"))

    assert (tmp_path / "storage" / "_blockfs_backup").exists()
    assert "Blockfs storage backed up" in caplog.messages

    with open(tmp_path / "storage" / "index.json", encoding="utf-8") as f:
        index = json.load(f)
        assert index["version"] == _LOCAL_STORAGE_VERSION
        assert index["migrations"] == []

    with open(
        tmp_path / "storage" / "_blockfs_backup" / "index.json", encoding="utf-8"
    ) as f:
        index = json.load(f)
        assert index["version"] == 0

    assert (
        tmp_path / "storage" / "_blockfs_backup" / "experiments" / "exp_dummy.txt"
    ).exists()
    assert (
        tmp_path / "storage" / "_blockfs_backup" / "ensembles" / "ens_dummy.txt"
    ).exists()
