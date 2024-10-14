import fnmatch
import os
import shutil
from unittest.mock import patch

import pytest

from ert.config import ErtConfig
from ert.storage import open_storage
from everest.config import EverestConfig
from everest.detached import generate_everserver_ert_config, start_server
from everest.simulator.everest_to_ert import _everest_to_ert_config_dict
from everest.strings import (
    DEFAULT_OUTPUT_DIR,
    DETACHED_NODE_DIR,
    OPTIMIZATION_OUTPUT_DIR,
)
from everest.suite import _EverestWorkflow
from everest.util import makedirs_if_needed


def test_that_one_experiment_creates_one_ensemble_per_batch(
    copy_math_func_test_data_to_tmp,
):
    config = EverestConfig.load_file("config_minimal.yml")
    workflow = _EverestWorkflow(config)
    assert workflow is not None

    workflow.start_optimization()

    batches = os.listdir(config.simulation_dir)
    ert_config = ErtConfig.with_plugins().from_dict(_everest_to_ert_config_dict(config))
    enspath = ert_config.ens_path

    with open_storage(enspath, mode="r") as storage:
        experiments = [*storage.experiments]
        assert len(experiments) == 1
        experiment = experiments[0]

        ensemble_names = {ens.name for ens in experiment.ensembles}
        assert ensemble_names == set(batches)


@pytest.mark.integration_test
@patch("ert.simulator.BatchSimulator.start", return_value=None)
def test_everest_output(start_mock, copy_mocked_test_data_to_tmp):
    config_folder = os.getcwd()
    config = EverestConfig.load_file("mocked_test_case.yml")
    everest_output_dir = config.output_dir

    (path, folders, files) = next(os.walk(config_folder))
    initial_folders = set(folders)
    initial_files = set(files)

    # Tests in this class used to fail when a callback was passed in
    # Use a callback just to see that everything works fine, even though
    # the callback does nothing
    def useless_cb(*args, **kwargs):
        pass

    _EverestWorkflow(config, optimization_callback=useless_cb)
    # Check the output folder is created when stating the optimization
    # in everest workflow
    assert DEFAULT_OUTPUT_DIR not in initial_folders
    assert os.path.exists(everest_output_dir)

    # Expected ropt and dakota output in a sub-folder from the expected output path
    assert OPTIMIZATION_OUTPUT_DIR in os.listdir(everest_output_dir)

    # If output folder is present because of test class setup, remove it
    if os.path.exists(everest_output_dir):
        shutil.rmtree(everest_output_dir)

    assert "storage" not in initial_folders
    assert DETACHED_NODE_DIR not in initial_folders
    ert_config = ErtConfig.with_plugins().from_dict(
        generate_everserver_ert_config(config)
    )
    makedirs_if_needed(config.output_dir, roll_if_exists=True)
    with open_storage(ert_config.ens_path, "w") as storage:
        start_server(config, ert_config, storage)
    start_mock.assert_called_once()

    (path, folders, files) = next(os.walk(config_folder))
    # Check we are looking at the config folder
    assert path == config_folder

    final_folders = set(folders)
    final_files = set(files)

    new_folders = final_folders.difference(initial_folders)
    # Check only the everest output folder was added
    assert list(new_folders) == [DEFAULT_OUTPUT_DIR]
    # Check no new files were created when starting the server
    assert len(final_files.difference(initial_files)) == 0
    # Check storage folder no longer created in the config folder
    assert "storage" not in final_folders
    makedirs_if_needed(config.output_dir, roll_if_exists=True)
    with open_storage(ert_config.ens_path, "w") as storage:
        start_server(config, ert_config, storage)
    assert start_mock.call_count == 2
    final_files = os.listdir(config_folder)

    # verify two everest_output dirs present
    assert len(fnmatch.filter(final_files, "everest_output*")) == 2


@patch("ert.simulator.BatchSimulator.start", return_value=None)
def test_save_running_config(start_mock, copy_math_func_test_data_to_tmp):
    file_name = "config_minimal.yml"
    config = EverestConfig.load_file(file_name)
    ert_config = ErtConfig.with_plugins().from_dict(
        generate_everserver_ert_config(config)
    )
    makedirs_if_needed(config.output_dir, roll_if_exists=True)
    with open_storage(ert_config.ens_path, "w") as storage:
        start_server(config, ert_config, storage)
    start_mock.assert_called_once()

    saved_config_path = os.path.join(config.output_dir, file_name)

    assert os.path.exists(saved_config_path)
    shutil.move(saved_config_path, os.path.join(os.getcwd(), "saved_config.yml"))

    new_config = EverestConfig.load_file("saved_config.yml")

    # Ignore different config names
    config.config_path = None
    new_config.config_path = None

    assert config == new_config
