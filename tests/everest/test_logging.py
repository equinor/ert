import os

import pytest

from ert.scheduler.event import FinishedEvent
from everest.config import EverestConfig, ServerConfig
from everest.detached import (
    start_server,
    wait_for_server,
)
from everest.util import makedirs_if_needed

CONFIG_FILE = "config_fm_failure.yml"


def string_exists_in_file(file_path, string):
    with open(file_path, "r", encoding="utf-8") as f:
        txt = f.read()
        return string in txt


@pytest.mark.timeout(60)  # Simulation might not finish
@pytest.mark.integration_test
@pytest.mark.xdist_group(name="starts_everest")
async def test_logging_setup(copy_math_func_test_data_to_tmp):
    async def server_running():
        while True:
            event = await driver.event_queue.get()
            if isinstance(event, FinishedEvent) and event.iens == 0:
                return

    everest_config = EverestConfig.load_file(CONFIG_FILE)

    makedirs_if_needed(everest_config.output_dir, roll_if_exists=True)
    driver = await start_server(everest_config, debug=True)
    try:
        wait_for_server(everest_config, 120)
    except SystemExit as e:
        raise e
    await server_running()

    everest_output_path = os.path.join(os.getcwd(), "everest_output")

    everest_logs_dir_path = everest_config.log_dir

    detached_node_dir = ServerConfig.get_detached_node_dir(everest_config.output_dir)
    endpoint_log_path = os.path.join(detached_node_dir, "endpoint.log")

    everest_log_path = os.path.join(everest_logs_dir_path, "everest.log")
    forward_model_log_path = os.path.join(everest_logs_dir_path, "forward_models.log")
    simulation_log_path = os.path.join(everest_logs_dir_path, "simulations.log")

    assert os.path.exists(everest_output_path)
    assert os.path.exists(everest_logs_dir_path)
    assert os.path.exists(forward_model_log_path)
    assert os.path.exists(simulation_log_path)
    assert os.path.exists(everest_log_path)
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
