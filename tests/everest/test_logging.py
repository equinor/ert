import os
from pathlib import Path

import pytest

from ert.scheduler.event import FinishedEvent
from everest.config import EverestConfig, ServerConfig
from everest.detached import (
    start_server,
    wait_for_server,
)
from everest.util import makedirs_if_needed

CONFIG_FILE = "config_fm_failure.yml"


def _string_exists_in_file(file_path, string):
    return string in Path(file_path).read_text(encoding="utf-8")


@pytest.mark.flaky(reruns=5)
@pytest.mark.timeout(70)  # Simulation might not finish
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
        wait_for_server(everest_config.output_dir, 60)
    except (SystemExit, RuntimeError) as e:
        raise e
    await server_running()

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

    assert _string_exists_in_file(everest_log_path, "everest DEBUG:")
    assert _string_exists_in_file(
        forward_model_log_path, "Exception: Failing simulation_2 by request!"
    )
    assert _string_exists_in_file(
        forward_model_log_path, "Exception: Failing simulation_2" " by request!"
    )

    endpoint_logs = Path(endpoint_log_path).read_text(encoding="utf-8")
    # Avoid cases where optimization finished before we get a chance to check that
    # the everest server has started
    if endpoint_logs:
        assert "everserver INFO: / entered from" in endpoint_logs
