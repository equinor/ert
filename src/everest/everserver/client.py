from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ert.scheduler import create_driver
from ert.scheduler.driver import Driver, FailedSubmit
from ert.scheduler.event import StartedEvent
from ert.services import ErtClient
from ert.trace import get_traceparent
from everest.config import EverestConfig
from everest.strings import (
    OPT_PROGRESS_ID,
    SIM_PROGRESS_ID,
)

if TYPE_CHECKING:
    from ert.run_models.event import EverestBatchResultEvent


# The methods in this file are typically called for the client side.
# Information from the client side is relatively uninteresting, so we show it in
# the default logger (stdout). Info from the server will be logged to the
# everest.log file instead
logger = logging.getLogger(__name__)


async def start_server(config: EverestConfig, logging_level: int) -> Driver:
    """Start an EVEREST server running the optimization defined in the config"""
    driver = create_driver(config.server.queue_system, poll_period=0.1)  # type: ignore
    try:
        args = [
            "--output-dir",
            str(config.output_dir),
            "--logging-level",
            str(logging_level),
            "--traceparent",
            str(get_traceparent()),
        ]
        poll_task = asyncio.create_task(driver.poll(), name="poll_task")
        await driver.submit(
            0, "everserver", *args, name=f"{Path(config.config_file).stem}-server"
        )
    except FailedSubmit as err:
        raise ValueError(f"Failed to submit Everserver with error: {err}") from err
    status = await driver.event_queue.get()
    if not isinstance(status, StartedEvent):
        poll_task.cancel()
        raise ValueError(f"Everserver not started as expected, got status: {status}")
    poll_task.cancel()
    logger.debug(
        f"Everserver started. Events left in driver queue: {driver.event_queue.qsize()}"
    )
    return driver


def get_opt_status_from_batch_result_event(
    event: EverestBatchResultEvent,
) -> dict[str, Any]:
    status = {
        "result_type": event.result_type,
        "batch": event.batch,
        "failures": event.failures,
    }
    if event.results and event.result_type == "FunctionResult":
        status.update(
            {
                "controls": event.results["controls"],
                "objective_value": event.results["total_objective_value"],
                "expected_objectives": event.results["objectives"],
            }
        )
    return status


def start_monitor(
    client: ErtClient,
    callback: Callable[[dict[str, Any]], None],
    experiment_id: str,
    polling_interval: float = 0.1,
) -> None:
    """
    Checks status on EVEREST server and calls callback when status changes

    Monitoring stops when the server stops answering.
    """
    from ert.run_models.event import (  # ruff: ignore[import-outside-top-level]
        EverestBatchResultEvent,
    )

    for event in client.iter_events(experiment_id, refresh_interval=polling_interval):
        if isinstance(event, EverestBatchResultEvent):
            callback({OPT_PROGRESS_ID: get_opt_status_from_batch_result_event(event)})
        else:
            callback({SIM_PROGRESS_ID: event})
