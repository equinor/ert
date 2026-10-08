from __future__ import annotations

import logging
import shutil
import sys
import traceback
from collections import defaultdict
from collections.abc import Callable, KeysView, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from _ert import ansi
from ert.ensemble_evaluator import (
    EnsembleSnapshot,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.ensemble_evaluator.event import EndEvent
from ert.services.ert_client import ErtClient
from everest.strings import EVEREST, OPT_PROGRESS_ID, SIM_PROGRESS_ID
from everest.util import format_list

if TYPE_CHECKING:
    from ert.run_models.event import EverestBatchResultEvent

JOB_SUCCESS = "Finished"
JOB_RUNNING = "Running"
JOB_FAILURE = "Failed"


def _get_max_width(sequence: Sequence[str] | KeysView[str]) -> int:
    return max(len(item) for item in sequence)


@dataclass
class JobProgress:
    name: str
    status: dict[str, list[int]] = field(
        default_factory=lambda: {
            JOB_RUNNING: [],  # contains running simulation numbers i.e [7,8,9]
            JOB_SUCCESS: [],  # contains successful simulation numbers i.e [0,1,3,4]
            JOB_FAILURE: [],  # contains failed simulation numbers i.e [5,6]
        }
    )
    errors: defaultdict[str, list[int]] = field(
        default_factory=lambda: defaultdict(list)
    )
    STATUS_COLOR: ClassVar = {
        JOB_RUNNING: ansi.BLUE,
        JOB_SUCCESS: ansi.GREEN,
        JOB_FAILURE: ansi.RED,
    }

    def progress_str(self, max_widths: dict[str, int]) -> str:
        string = []
        for state in [JOB_RUNNING, JOB_SUCCESS, JOB_FAILURE]:
            number_of_simulations = len(self.status[state])
            width = max_widths[state]
            color = self.STATUS_COLOR[state] if number_of_simulations else ansi.BLACK
            string.append(f"{color}{number_of_simulations:>{width}}{ansi.RESET}")
        return "/".join(string)


class _ServerMonitor:
    INDENT = 2
    FLOAT_FMT = ".5g"

    def __init__(self) -> None:
        self._clear_lines: int = 0
        self._last_reported_batch: int = -1
        self._last_reported_opt_progress: int = -1
        self._snapshots: dict[int, EnsembleSnapshot] = {}
        self._width = min(shutil.get_terminal_size(fallback=(78, 24)).columns, 100)

    def update(self, status: dict[str, Any]) -> None:
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            if OPT_PROGRESS_ID in status:
                opt_status = status[OPT_PROGRESS_ID]
                if opt_status:
                    msg = self._get_opt_progress_single_batch(opt_status)
                    if msg:
                        ansi.ansi_print(msg + "\n")
                        self._clear_lines = 0
            if SIM_PROGRESS_ID in status:
                match status[SIM_PROGRESS_ID]:
                    case EndEvent(msg=msg):
                        return
                    case FullSnapshotEvent(snapshot=snapshot, iteration=batch):
                        if snapshot is not None:
                            self._snapshots[batch] = snapshot
                    case (
                        SnapshotUpdateEvent(snapshot=snapshot, iteration=batch) as event
                    ):
                        if snapshot is not None:
                            batch_number = event.iteration
                            self._snapshots[batch_number].merge_snapshot(snapshot)
                            header = self._make_header(
                                f"Running forward models (Batch #{batch_number})",
                                ansi.BLUE,
                            )
                            summary = self._get_progress_summary(event.status_count)
                            job_states = self._get_job_states(
                                self._snapshots[batch_number]
                            )
                            msg = (
                                self._join_two_newlines_indent(
                                    (header, summary, job_states)
                                )
                                + "\n"
                            )
                            if batch == self._last_reported_batch:
                                self._clear()
                            ansi.ansi_print(msg)
                            self._clear_lines = len(msg.split("\n"))
                            self._last_reported_batch = max(
                                self._last_reported_batch, batch
                            )
        except Exception:
            logging.getLogger(EVEREST).debug(traceback.format_exc())

    def _get_opt_progress_batch(
        self, cli_monitor_data: dict[str, Any], batch: int, idx: int
    ) -> str:
        header = self._make_header(f"Optimization progress (Batch #{batch})")
        width = _get_max_width(cli_monitor_data["controls"][idx].keys())
        controls = self._join_one_newline_indent(
            [
                f"{name:>{width}}: {value:{self.FLOAT_FMT}}"
                for name, value in cli_monitor_data["controls"][idx].items()
            ]
        )
        expected_objectives = cli_monitor_data["expected_objectives"]
        width = _get_max_width(expected_objectives.keys())
        objectives = self._join_one_newline_indent(
            [
                f"{name:>{width}}: {value[idx]:{self.FLOAT_FMT}}"
                for name, value in expected_objectives.items()
            ]
        )
        objective_value = cli_monitor_data["objective_value"][idx]
        total_objective = (
            f"Total normalized objective: {objective_value:{self.FLOAT_FMT}}"
        )
        return self._join_two_newlines_indent(
            (header, controls, objectives, total_objective)
        )

    def _get_opt_progress_single_batch(self, cli_monitor_data: dict[str, Any]) -> str:
        batch: int = cli_monitor_data.get("batch", 0)
        if batch == self._last_reported_opt_progress:
            return ""

        lines = [self._make_header(f"Optimization progress (Batch #{batch})")]

        def mkline(data: dict[str, Any], width: int) -> str:
            width = _get_max_width(data.keys())
            return self._join_one_newline_indent(
                [
                    f"{name:>{width}}: {value:{self.FLOAT_FMT}}"
                    for name, value in data.items()
                ]
            )

        if cli_monitor_data.get("result_type") == "FunctionResult":
            if controls := cli_monitor_data.get("controls"):
                width = _get_max_width(controls.keys())
                lines.append(mkline(controls, width))
            if expected_objectives := cli_monitor_data.get("expected_objectives"):
                width = _get_max_width(expected_objectives.keys())
                lines.append(mkline(expected_objectives, width))
            if objective_value := cli_monitor_data.get("objective_value"):
                lines.append(
                    f"Total normalized objective: {objective_value:{self.FLOAT_FMT}}"
                )

        if failures := cli_monitor_data["failures"]:
            failed_lines = []
            if failed_functions := [r for r, p in failures.items() if -1 in p]:
                s = "s" if len(failed_functions) > 1 else ""
                failed_lines.append(
                    f"{ansi.RED}Failed function evaluation{s} for realization{s}: "
                    f"{format_list(failed_functions)}{ansi.RESET}"
                )
            for k, v in failures.items():
                if p := [item for item in v if item >= 0]:
                    s = "s" if len(p) > 1 else ""
                    failed_lines.append(
                        f"{ansi.RED}Failed perturbation{s} for realization {k}: "
                        f"{format_list(p)}{ansi.RESET}"
                    )
            if failed_lines:
                lines.append(self._join_one_newline_indent(failed_lines))

        self._last_reported_opt_progress = batch

        return self._join_two_newlines_indent(lines)

    @staticmethod
    def _get_progress_summary(status: dict[str, int]) -> str:
        colors = [
            ansi.BLACK,
            ansi.BLACK,
            ansi.BLUE if status.get("Running", 0) > 0 else ansi.BLACK,
            ansi.GREEN if status.get("Finished", 0) > 0 else ansi.BLACK,
            ansi.RED if status.get("Failed", 0) > 0 else ansi.BLACK,
        ]
        labels = ("Waiting", "Pending", "Running", "Finished", "Failed")
        values = [status.get(ls, 0) for ls in labels]
        return " | ".join(
            f"{color}{key}: {value}{ansi.RESET}"
            for color, key, value in zip(colors, labels, values, strict=False)
        )

    @classmethod
    def _get_job_states(cls, snapshot: EnsembleSnapshot) -> str:
        print_lines = []
        jobs_status = cls._get_jobs_status(snapshot)
        forward_model_messages = [
            v.get("message", "").replace(  # type: ignore[union-attr]
                "status from done callback:", "Forward model error:"
            )
            for v in snapshot.reals.values()
            if v.get("message")
        ]
        if jobs_status:
            max_widths = {
                state: _get_max_width(
                    [str(len(item.status[state])) for item in jobs_status]
                )
                for state in [JOB_RUNNING, JOB_SUCCESS, JOB_FAILURE]
            }
            width = _get_max_width([item.name for item in jobs_status])
            for job in jobs_status:
                print_lines.append(
                    f"{job.name:>{width}}: {job.progress_str(max_widths)}{ansi.RESET}"
                )
                if job.errors:
                    print_lines.extend(
                        [
                            f"{ansi.RED}{job.name:>{width}}: {err}{ansi.RESET}"
                            for err in job.errors
                        ]
                    )
                if forward_model_messages:
                    print_lines.extend(
                        [f"{ansi.RED} {message}" for message in forward_model_messages]
                    )
        return cls._join_one_newline_indent(print_lines)

    @staticmethod
    def _get_jobs_status(snapshot: EnsembleSnapshot) -> list[JobProgress]:
        job_progress = {}
        for (realization, job_idx), job in snapshot.get_all_fm_steps().items():
            assert "name" in job, "job name is missing"
            assert job["name"] is not None, "job name is None"
            name = job["name"]
            if job_idx not in job_progress:
                job_progress[job_idx] = JobProgress(name=name)
            assert "status" in job
            status = job["status"]
            if status in {JOB_RUNNING, JOB_SUCCESS, JOB_FAILURE}:
                job_progress[job_idx].status[status].append(int(realization))
            if error := job.get("error"):
                job_progress[job_idx].errors[error].append(int(realization))
        return list(job_progress.values())

    @classmethod
    def _join_one_newline_indent(cls, sequence: Sequence[str]) -> str:
        return ("\n" + " " * cls.INDENT).join(sequence)

    @classmethod
    def _join_two_newlines_indent(cls, sequence: Sequence[str]) -> str:
        return ("\n\n" + " " * cls.INDENT).join(sequence)

    @classmethod
    def _join_two_newlines(cls, sequence: Sequence[str]) -> str:
        return "\n\n".join(sequence)

    def _make_header(self, msg: str, color: str = ansi.BLACK) -> str:
        header = msg.center(len(msg) + 2).center(self._width, "=")
        return f"{color}{header}{ansi.RESET}"

    def _clear(self) -> None:
        if not sys.stdout.isatty():
            return
        for _ in range(self._clear_lines):
            print(ansi.CURSOR_UP, end=ansi.CLEAR_LINE)


def run_server_monitor(
    client: ErtClient,
    experiment_id: str,
) -> None:
    monitor = _ServerMonitor()
    start_monitor(client, callback=monitor.update, experiment_id=experiment_id)


def run_empty_server_monitor(
    client: ErtClient,
    experiment_id: str,
) -> None:
    start_monitor(client, callback=lambda _: None, experiment_id=experiment_id)


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
