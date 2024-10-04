import os

import pytest

from ert.config import ErtConfig
from ert.storage import open_storage
from everest.config import EverestConfig
from everest.detached import (
    context_stop_and_wait,
    generate_everserver_ert_config,
    start_server,
    wait_for_context,
    wait_for_server,
)
from everest.util import makedirs_if_needed

CONFIG_FILE = "config_fm_failure.yml"


def string_exists_in_file(file_path, string):
    with open(file_path, "r", encoding="utf-8") as f:
        txt = f.read()
        return string in txt


@pytest.mark.flaky(reruns=5)
@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
@pytest.mark.fails_on_macos_github_workflow
def test_logging_setup(copy_math_func_test_data_to_tmp):
    everest_config = EverestConfig.load_file(CONFIG_FILE)

    wait_for_context()
    ert_config = ErtConfig.with_plugins().from_dict(
        generate_everserver_ert_config(everest_config, True)
    )
    makedirs_if_needed(everest_config.output_dir, roll_if_exists=True)
    with open_storage(ert_config.ens_path, "w") as storage:
        start_server(everest_config, ert_config, storage)
        try:
            wait_for_server(everest_config, 120)
            wait_for_context()
        except SystemExit as e:
            context_stop_and_wait()
            raise e
    everest_output_path = os.path.join(os.getcwd(), "everest_output")

    everest_logs_dir_path = everest_config.log_dir

    detached_node_dir = everest_config.detached_node_dir
    endpoint_log_path = os.path.join(detached_node_dir, "endpoint.log")

    everest_log_path = os.path.join(everest_logs_dir_path, "everest.log")
    forward_model_log_path = os.path.join(everest_logs_dir_path, "forward_models.log")
    simulation_log_path = os.path.join(everest_logs_dir_path, "simulations.log")
    everest_server_stderr_path = os.path.join(
        everest_logs_dir_path, "everest_server.stderr.0"
    )
    everest_server_stdout_path = os.path.join(
        everest_logs_dir_path, "everest_server.stdout.0"
    )

    assert os.path.exists(everest_output_path)
    assert os.path.exists(everest_logs_dir_path)
    assert os.path.exists(forward_model_log_path)
    assert os.path.exists(simulation_log_path)
    assert os.path.exists(everest_log_path)
    assert os.path.exists(everest_server_stderr_path)
    assert os.path.exists(everest_server_stdout_path)
    assert os.path.exists(endpoint_log_path)

    assert string_exists_in_file(everest_log_path, "everest DEBUG:")
    assert string_exists_in_file(
        forward_model_log_path,
        "forward_models ERROR: Batch: 0 Realization: 0 Simulation: 2 "
        "Job: toggle_failure Failed Error: 0",
    )
    assert string_exists_in_file(
        forward_model_log_path, "Exception: Failing simulation_2" " by request!"
    )

    assert string_exists_in_file(
        endpoint_log_path,
        "everserver INFO: / entered from",
    )
