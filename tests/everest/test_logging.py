import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import yaml

from ert.scheduler.event import FinishedEvent
from everest.config import (
    EverestConfig,
    ServerConfig,
)
from everest.detached import start_server, wait_for_server
from everest.util import makedirs_if_needed


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

    config_yaml = await _read_file_as_yaml_async("config_minimal.yml")
    config_yaml["install_jobs"].append(
        {"name": "toggle_failure", "source": "jobs/FAIL_SIMULATION"}
    )
    config_yaml["forward_model"].append("toggle_failure --fail simulation_2")
    await _write_yaml_to_file_async(config_yaml, "config.yml")
    everest_config = EverestConfig.load_file("config.yml")

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


async def _read_file_as_yaml_async(path: str):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, _read_file_as_yaml, path)
    return result


async def _write_yaml_to_file_async(config_yaml: dict[str, Any], path: str):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, _write_yaml_to_file, config_yaml, path
        )
    return result


def _read_file_as_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def _write_yaml_to_file(config_yaml: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fout:
        yaml.dump(config_yaml, fout)
