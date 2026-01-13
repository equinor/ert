import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ert.storage import open_storage
from everest.bin.everest_script import everest_entry
from everest.config import EverestConfig
from tests.everest.utils import everest_config_with_defaults


@pytest.mark.xdist_group("math_func/config_minimal.yml")
@pytest.mark.integration_test
def test_that_one_experiment_creates_one_ensemble_per_batch(cached_example):
    _, config, _, _ = cached_example("math_func/config_minimal.yml")
    config = EverestConfig.load_file(config)
    batches = os.listdir(config.simulation_dir)
    with open_storage(config.storage_dir, mode="r") as storage:
        experiments = [*storage.experiments]
        assert len(experiments) == 1
        experiment = experiments[0]

        ensemble_names = {ens.name for ens in experiment.ensembles}
        assert ensemble_names == set(batches)


@patch(
    "everest.bin.everest_script.create_ertserver_client",
    side_effect=[TimeoutError(), MagicMock()],
)
@patch("everest.config.ServerConfig.get_server_context_from_conn_info")
@patch("everest.bin.everest_script.run_detached_monitor")
@patch("everest.bin.everest_script.wait_for_server")
@patch("everest.bin.everest_script.start_server")
@patch("everest.bin.everest_script.start_experiment")
def test_save_running_config(
    mock_start_experiment,
    mock_start_server,
    mock_wait_for_server,
    mock_run_detached_monitor,
    mock_get_server_context,
    mock_start_session,
    change_to_tmpdir,
):
    """Test everest detached, when an optimization has already run"""

    Path("config.yml").touch()
    config = everest_config_with_defaults(
        config_path="./config.yml", environment={"random_seed": 12345}
    )
    config.write_to_file("config.yml")

    everest_entry(["config.yml", "--skip-prompt"])
    saved_config_path = os.path.join(config.output_dir, "config.yml")

    assert os.path.exists(saved_config_path)
    shutil.move(saved_config_path, os.path.join(os.getcwd(), "saved_config.yml"))

    new_config = EverestConfig.load_file("saved_config.yml")

    # Ignore different config names
    old_config_dict = {**config.model_dump(exclude_none=True), "config_path": None}
    new_config_dict = {**new_config.model_dump(exclude_none=True), "config_path": None}

    assert old_config_dict == new_config_dict
