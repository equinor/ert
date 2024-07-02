import os
import shutil

import numpy as np
import pytest
import xarray as xr
from packaging import version

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
            str(experiment.response_configuration) + "\n", "responses"
        )

        assert ensemble.has_combined_response_dataset("gen_data")
        assert ensemble.has_combined_response_dataset("summary")

        summary_data = ensemble.load_responses(
            "summary",
            tuple(ensemble.get_realization_with_responses("summary")),
        )
        snapshot.assert_match(
            summary_data.to_dataframe(dim_order=["realization", "name", "time"])
            .transform(np.sort)
            .to_csv(),
            "summary_data",
        )

        expect_dir = {}
        for group, ds in experiment.observations["gen_data"].groupby(
            "obs_name", squeeze=True
        ):
            expect_dir[group] = (
                ds.drop("obs_name").squeeze("name", drop=True).to_dataframe().to_csv()
            )

        snapshot.assert_match_dir(expect_dir, "observations")


@pytest.mark.usefixtures("copy_shared")
@pytest.mark.parametrize(
    "ert_version",
    [
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
    tmp_path, block_storage_path, snapshot, monkeypatch, ert_version
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

        ens_dir_contents = set(os.listdir(ensemble_path))
        assert {
            "BPR.nc",
            "PORO.nc",
            "index.json",
            "summary.nc",
        }.issubset(ens_dir_contents)

        assert "TOP.nc" not in ens_dir_contents

        with pytest.raises(KeyError):
            ensembles[0].load_responses("GEN")


@pytest.mark.usefixtures("copy_shared")
@pytest.mark.parametrize(
    "ert_version",
    [
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
def test_that_storage_always_has_state_map_after_migrations(
    tmp_path, block_storage_path, snapshot, monkeypatch, ert_version
):
    storage_path = tmp_path / "all_data_types" / f"storage-{ert_version}"
    shutil.copytree(
        block_storage_path / f"all_data_types/storage-{ert_version}",
        storage_path,
    )
    [ensemble_id] = os.listdir(storage_path / "ensembles")

    ensemble_path = storage_path / "ensembles" / ensemble_id

    ds_bpr1 = xr.open_dataset((ensemble_path / "realization-1/BPR.nc"))
    ds_gen1 = xr.open_dataset((ensemble_path / "realization-1/GEN.nc"))

    ensembles_to_edit = {1, 3, 6, 8}
    for i in ensembles_to_edit:
        real_dir = ensemble_path / f"realization-{i}"
        os.remove(real_dir / "BPR.nc")
        os.remove(real_dir / "GEN.nc")

        assert not os.path.exists(real_dir / "BPR.nc")
        assert not os.path.exists(real_dir / "GEN.nc")

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
        [ensemble] = ensembles

        assert ensemble._realization_states is not None
        assert os.path.exists(ensemble_path / "state_map.json")

        ensemble_param_mask = ensemble.get_realization_mask_with_parameters()
        ensemble_response_mask = ensemble.get_realization_mask_with_responses()
        print(f"ensemble_param_mask=[{ensemble_param_mask}]")
        print(f"ensemble_response_mask=[{ensemble_response_mask}]")
        for i in range(ensemble.ensemble_size):
            assert ensemble_param_mask[i] == (i not in ensembles_to_edit)
            assert ensemble_response_mask[i] == (i not in ensembles_to_edit)

        for i in ensembles_to_edit:
            _ds_gen1 = ds_gen1.copy(deep=True)
            _ds_gen1.coords["realization"] = [i]

            ensemble.save_response("GEN", _ds_gen1, i)

            _ds_bpr1 = ds_bpr1.copy(deep=True)
            _ds_bpr1.coords["realizations"] = [i]
            ensemble.save_parameters("BPR", i, _ds_bpr1)

        ensemble.refresh_statemap()
        ensemble.unify_parameters()
        ensemble.unify_responses()

        ensemble_param_mask_after_save = ensemble.get_realization_mask_with_parameters()
        ensemble_response_mask_after_save = (
            ensemble.get_realization_mask_with_responses()
        )

        assert all(ensemble_param_mask_after_save)
        assert all(ensemble_response_mask_after_save)
