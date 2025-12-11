import datetime
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import pytest

from ert.analysis import ErtAnalysisError, smoother_update
from ert.config import ErtConfig, ESSettings, ObservationSettings
from ert.plugins import get_site_plugins
from ert.storage import RealizationStorageState, open_storage
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


@pytest.fixture()
def copy_shared_design(tmp_path, block_storage_path):
    shutil.copytree(
        block_storage_path / "design_poly",
        tmp_path / "design_poly",
    )
    yield tmp_path / "design_poly"


@pytest.mark.parametrize(
    "ert_version",
    [
        "14.2",
    ],
)
def test_migration_to_genkw_with_polars_and_design_matrix(
    copy_shared_design, ert_version, snapshot
):
    # we need to make a dummy ert config to open storage
    local_storage_set_ert_config(ErtConfig.from_file_contents("NUM_REALIZATIONS 1\n"))
    storage_path = copy_shared_design / f"version-{ert_version}"
    with open_storage(storage_path, "w") as storage:
        experiments = list(storage.experiments)
        assert len(experiments) == 1
        experiment = experiments[0]
        ensembles = list(experiment.ensembles)
        assert len(ensembles) == 1
        ensemble = ensembles[0]
        df = ensemble.load_parameters("DESIGN_MATRIX")
        assert isinstance(df, pl.DataFrame)
        assert dict(df.schema) == dict(
            pl.Schema(
                {
                    "a": pl.Int64,
                    "category": pl.String,
                    "b": pl.Int64,
                    "c": pl.Int64,
                    "realization": pl.Int64,
                }
            )
        )
        snapshot.assert_match(
            orjson.dumps(df.to_dicts(), option=orjson.OPT_INDENT_2)
            .decode("utf-8")
            .strip()
            + "\n",
            "design_matrix_snapshot.json",
        )


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_shared")
@pytest.mark.parametrize(
    "ert_version",
    [
        "15.0.2",
        "14.6.4",
        "13.0.5",
        "12.1.2",
        "11.1.9",
        "11.1.8",
        "11.0.8",
        "10.3.1",
        "10.2.8",
        "10.1.3",
        "10.0.3",
        "9.0.17",
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
    site_plugins = get_site_plugins()
    ert_config = ErtConfig.with_plugins(site_plugins).from_file("config.ert")

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

        assert all(
            "has_finalized_keys" in config
            for config in experiment.response_info.values()
        )

        with open(
            experiment._path / experiment._responses_file, "w", encoding="utf-8"
        ) as f:
            json.dump(
                {k: v.model_dump(mode="json") for k, v in response_config.items()},
                f,
                default=str,
                indent=2,
            )

        assert experiment.parameter_configuration["PORO"].ertbox_params.nx == 2
        assert experiment.parameter_configuration["PORO"].ertbox_params.ny == 3
        assert experiment.parameter_configuration["PORO"].ertbox_params.nz == 4

        assert experiment.templates_configuration == [("\nBPR:<BPR>\n", "params.txt")]
        df = ensemble.load_parameters("BPR")
        assert isinstance(df, pl.DataFrame)
        assert dict(df.schema) == {"BPR": pl.Float64, "realization": pl.Int64}
        assert df["realization"].to_list() == list(range(ensemble.ensemble_size))
        snapshot.assert_match(
            json.dumps(
                {
                    k: v.model_dump(mode="json")
                    for k, v in sorted(experiment.parameter_configuration.items())
                },
                default=str,
                indent=2,
            )
            + "\n",
            "parameters",
        )
        snapshot.assert_match(
            json.dumps(
                {
                    k: v.model_dump(mode="json")
                    for k, v in sorted(experiment.response_configuration.items())
                },
                default=str,
                indent=2,
            )
            + "\n",
            "responses",
        )

        summary_data = ensemble.load_responses(
            "summary",
            tuple(ensemble.get_realization_list_with_responses()),
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

        gen_data = ensemble.load_responses(
            "gen_data", tuple(range(ensemble.ensemble_size))
        )
        snapshot.assert_match(
            gen_data.sort(["realization", "response_key", "report_step", "index"])
            .to_pandas()
            .set_index(["realization", "response_key", "report_step", "index"])
            .to_csv(),
            "gen_data",
        )

        assert ensemble.experiment._has_finalized_response_keys("summary")
        assert ensemble.experiment._has_finalized_response_keys("gen_data")
        ensemble.save_response("summary", ensemble.load_responses("summary", (0,)), 0)
        assert ensemble.experiment._has_finalized_response_keys("summary")
        assert ensemble.experiment.response_type_to_response_keys["summary"] == ["FOPR"]


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_shared")
@pytest.mark.parametrize(
    "ert_version",
    [
        "11.1.8",
        "11.0.8",
        "10.3.1",
        "10.2.8",
        "10.1.3",
        "10.0.3",
        "9.0.17",
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

        gen_data_file = next(
            file for file in os.listdir(real_dir) if "gen" in file.lower()
        )
        os.remove(real_dir / gen_data_file)

    monkeypatch.chdir(tmp_path / "all_data_types")
    site_plugins = get_site_plugins()
    ert_config = ErtConfig.with_plugins(site_plugins).from_file("config.ert")

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


@pytest.mark.integration_test
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


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_shared")
@pytest.mark.parametrize(
    "ert_version",
    [
        "10.3.1",
    ],
)
def test_that_manual_update_from_migrated_storage_works(
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
    site_plugins = get_site_plugins()
    ert_config = ErtConfig.with_plugins(site_plugins).from_file("config.ert")

    local_storage_set_ert_config(ert_config)
    # To make sure all tests run against the same snapshot
    snapshot.snapshot_dir = snapshot.snapshot_dir.parent
    with open_storage(f"storage-{ert_version}", "w") as storage:
        experiments = list(storage.experiments)
        assert len(experiments) == 1
        experiment = experiments[0]
        ensembles = list(experiment.ensembles)
        assert len(ensembles) == 1
        prior_ens = ensembles[0]

        assert set(experiment.observations["gen_data"].schema.items()) == {
            ("index", pl.UInt16),
            ("observation_key", pl.String),
            ("observations", pl.Float32),
            ("report_step", pl.UInt16),
            ("response_key", pl.String),
            ("std", pl.Float32),
        }

        assert set(experiment.observations["summary"].schema.items()) == {
            ("observation_key", pl.String),
            ("observations", pl.Float32),
            ("response_key", pl.String),
            ("std", pl.Float32),
            ("time", pl.Datetime(time_unit="ms")),
        }

        prior_gendata = prior_ens.load_responses(
            "gen_data", tuple(range(prior_ens.ensemble_size))
        )
        prior_smry = prior_ens.load_responses(
            "summary", tuple(range(prior_ens.ensemble_size))
        )

        assert set(prior_gendata.schema.items()) == {
            ("response_key", pl.String),
            ("index", pl.UInt16),
            ("realization", pl.UInt16),
            ("report_step", pl.UInt16),
            ("values", pl.Float32),
        }

        assert set(prior_smry.schema.items()) == {
            ("response_key", pl.String),
            ("time", pl.Datetime(time_unit="ms")),
            ("realization", pl.UInt16),
            ("values", pl.Float32),
        }

        posterior_ens = storage.create_ensemble(
            prior_ens.experiment_id,
            ensemble_size=prior_ens.ensemble_size,
            iteration=1,
            name="posterior",
            prior_ensemble=prior_ens,
        )

        with pytest.raises(
            ErtAnalysisError, match="No active observations for update step"
        ):
            smoother_update(
                prior_ens,
                posterior_ens,
                list(experiment.observation_keys),
                list(ert_config.ensemble_config.parameters),
                ObservationSettings(),
                ESSettings(),
            )


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_shared")
@pytest.mark.parametrize(
    "ert_version",
    [
        "15.0.2",
        "14.6.4",
        "13.0.5",
        "12.1.2",
        "11.1.9",
        "11.1.8",
        "10.3.1",
        "10.0.3",
        "9.0.17",
    ],
)
def test_migrate_storage_with_no_responses(
    tmp_path,
    block_storage_path,
    monkeypatch,
    ert_version,
):
    storage_path = tmp_path / "all_data_types" / f"storage-{ert_version}"
    shutil.copytree(
        block_storage_path / f"all_data_types/storage-{ert_version}",
        storage_path,
    )
    [ensemble_id] = os.listdir(storage_path / "ensembles")

    # Remove all realization-*/TOP.nc, and only some realization-*/BPC.nc
    for real_dir in (storage_path / "ensembles" / ensemble_id).glob("realization-*"):
        gen_data_file = next(
            file for file in os.listdir(real_dir) if "gen" in file.lower()
        )

        os.remove(real_dir / gen_data_file)

        summary_file = next(
            file for file in os.listdir(real_dir) if "summary" in file.lower()
        )

        os.remove(real_dir / summary_file)

    monkeypatch.chdir(tmp_path / "all_data_types")
    site_plugins = get_site_plugins()
    ert_config = ErtConfig.with_plugins(site_plugins).from_file("config.ert")

    local_storage_set_ert_config(ert_config)

    open_storage(f"storage-{ert_version}", "w")


@pytest.mark.integration_test
@pytest.mark.usefixtures("copy_shared")
@pytest.mark.parametrize(
    "ert_version",
    [
        "15.0.2",
        "14.6.4",
        "13.0.5",
        "12.1.2",
        "11.1.9",
        "11.1.8",
        "10.3.1",
        "10.0.3",
        "9.0.17",
    ],
)
def test_that_storages_with_failed_realizations_are_migrated_without_errors(
    tmp_path,
    block_storage_path,
    monkeypatch,
    ert_version,
):
    storage_path = tmp_path / "all_data_types" / f"storage-{ert_version}"
    shutil.copytree(
        block_storage_path / f"all_data_types/storage-{ert_version}",
        storage_path,
    )
    monkeypatch.chdir(tmp_path / "all_data_types")

    # Inject a failed realization
    ensembles_dir = Path(f"storage-{ert_version}/ensembles")
    # There are no missing reals, so we assume the index corresponds
    # to the real nr
    realization_dirs = sorted(ensembles_dir.glob("*/realization-*"))

    # Add failures to some of them
    failures = [
        (i, failure_json, realization_state)
        for i, failure_json, realization_state in [
            (
                0,
                {
                    "type": 1,
                    "message": "this is undefined",
                    "time": datetime.datetime(2000, 1, 5).isoformat(),
                },
                RealizationStorageState.UNDEFINED,
            ),
            (
                3,
                {
                    "type": 8,
                    "message": "this is failure",
                    "time": datetime.datetime(2000, 1, 5).isoformat(),
                },
                RealizationStorageState.FAILURE_IN_CURRENT,
            ),
            (
                8,
                {
                    "type": 16,
                    "message": "this is parent failure",
                    "time": datetime.datetime(2000, 1, 5).isoformat(),
                },
                RealizationStorageState.FAILURE_IN_PARENT,
            ),
        ]
    ]

    for i, failure_json, _ in failures:
        (realization_dirs[i] / "error.json").write_text(json.dumps(failure_json))

    ert_config = ErtConfig.with_plugins(get_site_plugins()).from_file("config.ert")

    local_storage_set_ert_config(ert_config)

    with open_storage(f"storage-{ert_version}", "w") as storage:
        ensemble = next(storage.ensembles)
        realization_states = ensemble.get_ensemble_state()
        for i, _, realization_state in failures:
            assert realization_state in realization_states[i]
            assert ensemble.get_failure(i).type == realization_state
