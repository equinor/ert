import os
from pathlib import Path

import pytest

from everest.bin.main import start_everest
from everest.config import (
    EverestConfig,
    ServerConfig,
)
from everest.config.install_job_config import InstallJobConfig


@pytest.mark.timeout(240)  # Simulation might not finish
@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
def test_logging_setup(copy_math_func_test_data_to_tmp):
    everest_config = EverestConfig.load_file("config_minimal.yml")
    everest_config.forward_model.append("toggle_failure --fail simulation_2")
    everest_config.install_jobs.append(
        InstallJobConfig(name="toggle_failure", source="jobs/FAIL_SIMULATION")
    )
    everest_config.optimization.min_pert_success = 1
    everest_config.optimization.max_iterations = 1
    everest_config.optimization.min_realizations_success = 1
    everest_config.optimization.perturbation_num = 2

    # start_server() loads config based on config_path, so we need to actually
    # overwrite it
    everest_config.dump("config_minimal.yml")
    start_everest(["everest", "run", "config_minimal.yml", "--skip-prompt"])

    everest_output_path = os.path.join(os.getcwd(), "everest_output")
    everest_logs_dir_path = everest_config.log_dir
    detached_node_dir = ServerConfig.get_detached_node_dir(everest_config.output_dir)
    endpoint_log_path = os.path.join(detached_node_dir, "endpoint.log")
    everest_log_path = os.path.join(everest_logs_dir_path, "everest.log")
    forward_model_log_path = os.path.join(everest_logs_dir_path, "forward_models.log")

    assert os.path.exists(everest_output_path)
    assert os.path.exists(everest_logs_dir_path)
    assert os.path.exists(forward_model_log_path)
    assert os.path.exists(everest_log_path)
    assert os.path.exists(endpoint_log_path)

    assert "everest.detached.jobs.everserver INFO: Output directory:" in Path(
        everest_log_path
    ).read_text(encoding="utf-8")
    assert "Process exited with status code 1" in Path(
        forward_model_log_path
    ).read_text(encoding="utf-8")

    endpoint_logs = Path(endpoint_log_path).read_text(encoding="utf-8")
    # Avoid cases where optimization finished before we get a chance to check that
    # the everest server has started
    if endpoint_logs:
        assert "everserver INFO: / entered from" in endpoint_logs
