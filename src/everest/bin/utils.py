import argparse
import json
import logging
import os
import sys
import traceback
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from textwrap import dedent
from typing import Any, ClassVar

import colorama
import yaml
from colorama import Fore

from ert.config import QueueSystem
from ert.ensemble_evaluator import (
    EnsembleSnapshot,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.logging import LOGGING_CONFIG
from everest.config import EverestConfig
from everest.config.server_config import ServerConfig
from everest.detached import (
    ExperimentState,
    everserver_status,
    server_is_running,
    start_monitor,
    stop_server,
    wait_for_server_to_stop,
)
from everest.plugins.everest_plugin_manager import EverestPluginManager
from everest.simulator import JOB_FAILURE, JOB_RUNNING, JOB_SUCCESS
from everest.strings import EVEREST, OPT_PROGRESS_ID, SIM_PROGRESS_ID
from everest.util import makedirs_if_needed


def cleanup_logging() -> None:
    os.environ.pop("ERT_LOG_DIR", None)


@contextmanager
def setup_logging(options: argparse.Namespace) -> Generator[None, None, None]:
    if isinstance(options.config, EverestConfig):
        makedirs_if_needed(options.config.output_dir, roll_if_exists=False)
        log_dir = Path(options.config.output_dir) / "logs"
    else:
        # `everest branch` gives a tuple object here.
        log_dir = Path("logs")

    try:
        log_dir.mkdir(exist_ok=True)
    except PermissionError as err:
        sys.exit(str(err))
    try:
        os.environ["ERT_LOG_DIR"] = str(log_dir)

        with open(LOGGING_CONFIG, encoding="utf-8") as log_conf_file:
            config_dict = yaml.safe_load(log_conf_file)
            if config_dict:
                for handler_name, handler_config in config_dict["handlers"].items():
                    if handler_name == "file":
                        handler_config["filename"] = "everest-log.txt"
                    if "ert.logging.TimestampedFileHandler" in handler_config.values():
                        handler_config["config_filename"] = ""
                        if isinstance(options.config, EverestConfig):
                            handler_config["config_filename"] = (
                                options.config.config_path.name
                            )
                        else:
                            # `everest branch`
                            handler_config["config_filename"] = options.config[0]
                logging.config.dictConfig(config_dict)

        if "debug" in options and options.debug:
            root_logger = logging.getLogger()
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            root_logger.addHandler(handler)

        plugin_manager = EverestPluginManager()
        plugin_manager.add_log_handle_to_root()
        plugin_manager.add_span_processor_to_trace_provider()
        yield
    finally:
        cleanup_logging()


def handle_keyboard_interrupt(signum: int, _: Any, options: argparse.Namespace) -> None:
    print("\n" + "=" * 80)
    if options.config.server_queue_system == QueueSystem.LOCAL:
        server_context = ServerConfig.get_server_context(options.config.output_dir)
        if server_is_running(*server_context):
            print(
                f"KeyboardInterrupt (ID: {signum}) has been caught. \n"
                "You are running locally. \n"
                "The optimization will be stopped and the program will exit..."
            )
            stop_server(server_context)
            wait_for_server_to_stop(server_context, timeout=10)

    else:
        print(f"KeyboardInterrupt (ID: {signum}) has been caught. Program will exit...")
        config_file = options.config.config_file
        print(
            "You are running in detached mode.\n"
            "To monitor the running optimization use command:\n"
            f"  `everest monitor {config_file}`\n"
            "To kill the running optimization use command:\n"
            f"  `everest kill {config_file}`"
        )
    print("=" * 80)
    sys.tracebacklimit = 0
    sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa SIM115
    sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa SIM115
    sys.exit()


def _get_max_width(sequence: list[Any]) -> int:
    return max(len(item) for item in sequence)


def _format_list(values: Sequence[int]) -> str:
    """Formats a sequence of integers into a comma separated string of ranges.

    For instance: {1, 3, 4, 5, 7, 8, 10} -> "1, 3-5, 7-8, 10"
    """
    grouped = (
        tuple(y for _, y in x)
        for _, x in groupby(enumerate(sorted(values)), lambda x: x[0] - x[1])
    )
    return ", ".join(
        (
            "-".join([str(sub_group[0]), str(sub_group[-1])])
            if len(sub_group) > 1
            else str(sub_group[0])
        )
        for sub_group in grouped
    )


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
        JOB_RUNNING: Fore.BLUE,
        JOB_SUCCESS: Fore.GREEN,
        JOB_FAILURE: Fore.RED,
    }

    def _status_string(self, max_widths: dict[str, int]) -> str:
        string = []
        for state in [JOB_RUNNING, JOB_SUCCESS, JOB_FAILURE]:
            number_of_simulations = len(self.status[state])
            width = max_widths[state]
            color = self.STATUS_COLOR[state] if number_of_simulations else Fore.BLACK
            string.append(f"{color}{number_of_simulations:>{width}}{Fore.RESET}")
        return "/".join(string)

    def progress_str(self, max_widths: dict[str, int]) -> str:
        msg = ""
        for state in [JOB_SUCCESS, JOB_FAILURE]:
            simulations_list = _format_list(self.status[state])
            width = _get_max_width([simulations_list])
            if width > 0:
                color = self.STATUS_COLOR[state]
                msg += f" | {color}{state}: {simulations_list:<{width}}{Fore.RESET}"

        return self._status_string(max_widths) + msg


class _DetachedMonitor:
    WIDTH = 78
    INDENT = 2
    FLOAT_FMT = ".5g"

    def __init__(self) -> None:
        self._clear_lines: int = 0
        self._batches_done = set[int]()
        self._last_reported_batch: int = -1
        colorama.init(autoreset=True)
        self._snapshots: dict[int, EnsembleSnapshot] = {}

    def update(self, status: dict[str, Any]) -> None:
        try:
            if OPT_PROGRESS_ID in status:
                opt_status = status[OPT_PROGRESS_ID]
                if opt_status:
                    msg = self._get_opt_progress_single_batch(opt_status)
                    print(msg + "\n")
                    self._clear_lines = 0
            if SIM_PROGRESS_ID in status:
                match status[SIM_PROGRESS_ID]:
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
                                Fore.BLUE,
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
                            print(msg)
                            self._clear_lines = len(msg.split("\n"))
                            self._last_reported_batch = max(
                                self._last_reported_batch, batch
                            )
        except Exception:
            logging.getLogger(EVEREST).debug(traceback.format_exc())

    def get_opt_progress(self, context_status: dict[str, Any]) -> tuple[str, int]:
        cli_monitor_data = context_status["cli_monitor_data"]
        messages = []
        first_batch = -1
        for idx, batch in enumerate(cli_monitor_data["batches"]):
            if batch not in self._batches_done:
                if first_batch < 0:
                    first_batch = batch
                self._batches_done.add(batch)
                msg = self._get_opt_progress_batch(cli_monitor_data, batch, idx)
                messages.append(msg)
        return self._join_two_newlines(messages), first_batch

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
        header = self._make_header(f"Optimization progress (Batch #{batch})")
        width = _get_max_width(cli_monitor_data["controls"].keys())
        controls = self._join_one_newline_indent(
            [
                f"{name:>{width}}: {value:{self.FLOAT_FMT}}"
                for name, value in cli_monitor_data["controls"].items()
            ]
        )
        expected_objectives = cli_monitor_data["expected_objectives"]
        width = _get_max_width(expected_objectives.keys())
        objectives = self._join_one_newline_indent(
            [
                f"{name:>{width}}: {value:{self.FLOAT_FMT}}"
                for name, value in expected_objectives.items()
            ]
        )
        objective_value = cli_monitor_data["objective_value"]
        total_objective = (
            f"Total normalized objective: {objective_value:{self.FLOAT_FMT}}"
        )
        return self._join_two_newlines_indent(
            (header, controls, objectives, total_objective)
        )

    @staticmethod
    def _get_progress_summary(status: dict[str, int]) -> str:
        colors = [
            Fore.BLACK,
            Fore.BLACK,
            Fore.BLUE if status.get("Running", 0) > 0 else Fore.BLACK,
            Fore.GREEN if status.get("Finished", 0) > 0 else Fore.BLACK,
            Fore.RED if status.get("Failed", 0) > 0 else Fore.BLACK,
        ]
        labels = ("Waiting", "Pending", "Running", "Finished", "Failed")
        values = [status.get(ls, 0) for ls in labels]
        return " | ".join(
            f"{color}{key}: {value}{Fore.RESET}"
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
            for _, v in snapshot.reals.items()
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
                    f"{job.name:>{width}}: {job.progress_str(max_widths)}{Fore.RESET}"
                )
                if job.errors:
                    print_lines.extend(
                        [
                            f"{Fore.RED}{job.name:>{width}}: Failed: {err}, "
                            f"realizations: {_format_list(job.errors[err])}{Fore.RESET}"
                            for err in job.errors
                        ]
                    )
                if forward_model_messages:
                    print_lines.extend(
                        [f"{Fore.RED} {message}" for message in forward_model_messages]
                    )
        return cls._join_one_newline_indent(print_lines)

    @staticmethod
    def _get_jobs_status(snapshot: EnsembleSnapshot) -> list[JobProgress]:
        job_progress = {}
        for (realization, job_idx), job in snapshot.get_all_fm_steps().items():
            assert "name" in job and job["name"] is not None, "job name is missing"
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

    @classmethod
    def _make_header(cls, msg: str, color: str = Fore.BLACK) -> str:
        header = msg.center(len(msg) + 2).center(cls.WIDTH, "=")
        return f"{color}{header}{Fore.RESET}"

    def _clear(self) -> None:
        for _ in range(self._clear_lines):
            print(colorama.Cursor.UP(), end=colorama.ansi.clear_line())


def run_detached_monitor(server_context: tuple[str, str, tuple[str, str]]) -> None:
    monitor = _DetachedMonitor()
    start_monitor(server_context, callback=monitor.update)


def run_empty_detached_monitor(
    server_context: tuple[str, str, tuple[str, str]],
) -> None:
    start_monitor(server_context, callback=lambda _: None)


def report_on_previous_run(
    config_file: str,
    everserver_status_path: str,
    optimization_output_dir: str,
) -> None:
    server_state = everserver_status(everserver_status_path)
    if server_state["status"] == ExperimentState.failed:
        error_msg = server_state["message"]
        print(
            f"Optimization run failed, with error: {error_msg}\n"
            "To re-run optimization case use command:\n"
            f"`  everest run --new-run {config_file}`\n"
        )
    else:
        print(
            f"Optimization completed.\n"
            "\nTo re-run the optimization use command:\n"
            f"  `everest run --new-run {config_file}`\n"
            f"Results are stored in {optimization_output_dir}"
        )


def _read_user_preferences(user_info_path: Path) -> dict[str, dict[str, Any]]:
    try:
        if user_info_path.exists():
            with open(user_info_path, encoding="utf-8") as f:
                return json.load(f)

        user_info = {EVEREST: {"show_scaling_warning": True}}
        with open(user_info_path, mode="w", encoding="utf-8") as f:
            json.dump(user_info, f, ensure_ascii=False, indent=4)
    except json.decoder.JSONDecodeError:
        return {EVEREST: {}}
    else:
        return user_info


def show_scaled_controls_warning() -> None:
    user_info_path = Path(os.getenv("HOME", "")) / ".ert"
    user_info = _read_user_preferences(user_info_path)
    everest_pref = user_info.get(EVEREST, {})

    if not everest_pref.get("show_scaling_warning", True):
        return

    user_input = input(
        dedent("""
        From Everest version: 14.0.3, Everest will output auto-scaled control values.
        Control values should now be specified in real-world units instead of the
        optimizer's internal scale. The 'scaled_range' property can still be used
        to configure the optimizer's range for each control.

        [Enter] to continue.
        [  Y  ] to stop showing this message again.
        [  N  ] to abort.
        """)
    ).lower()
    match user_input:
        case "y":
            everest_pref["show_scaling_warning"] = False
            try:
                with open(user_info_path, mode="w", encoding="utf-8") as f:
                    json.dump(user_info, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logging.getLogger(EVEREST).error(str(e))
        case "n":
            raise SystemExit(0)
