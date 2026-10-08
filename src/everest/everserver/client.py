from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ert.services import ErtClient
from everest.strings import (
    OPT_PROGRESS_ID,
    SIM_PROGRESS_ID,
)

if TYPE_CHECKING:
    from ert.run_models.event import EverestBatchResultEvent


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
