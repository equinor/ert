import os
import shutil
from unittest.mock import patch

import pytest

from ert.config import ErtConfig
from ert.storage import open_storage
from everest.bin.everest_script import everest_entry
from everest.config import EverestConfig
from everest.detached import ServerStatus
from everest.simulator.everest_to_ert import _everest_to_ert_config_dict


@pytest.mark.xdist_group("math_func/config_minimal.yml")
def test_that_one_experiment_creates_one_ensemble_per_batch(cached_example):
    _, config, _ = cached_example("math_func/config_minimal.yml")
    config = EverestConfig.load_file(config)

    batches = os.listdir(config.simulation_dir)
    ert_config = ErtConfig.with_plugins().from_dict(_everest_to_ert_config_dict(config))
    enspath = ert_config.ens_path

    with open_storage(enspath, mode="r") as storage:
        experiments = [*storage.experiments]
        assert len(experiments) == 1
        experiment = experiments[0]

        ensemble_names = {ens.name for ens in experiment.ensembles}
        assert ensemble_names == set(batches)


@patch("everest.bin.everest_script.server_is_running", return_value=False)
@patch("everest.bin.everest_script.run_detached_monitor")
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch("everest.bin.everest_script.start_experiment")
@patch(
    "everest.bin.everest_script.everserver_status",
    return_value={"status": ServerStatus.never_run, "message": None},
)
def test_save_running_config(_, _1, _2, _3, _4, _5, copy_math_func_test_data_to_tmp):
    """Test everest detached, when an optimization has already run"""
    # optimization already run, notify the user
    file_name = "config_minimal.yml"
    config = EverestConfig.load_file(file_name)
    everest_entry([file_name])
    saved_config_path = os.path.join(config.output_dir, file_name)

    assert os.path.exists(saved_config_path)
    shutil.move(saved_config_path, os.path.join(os.getcwd(), "saved_config.yml"))

    new_config = EverestConfig.load_file("saved_config.yml")

    # Ignore different config names
    old_config_dict = {**config.model_dump(exclude_none=True), "config_path": None}
    new_config_dict = {**new_config.model_dump(exclude_none=True), "config_path": None}

    assert old_config_dict == new_config_dict
