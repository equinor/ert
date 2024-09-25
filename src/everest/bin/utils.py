import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from itertools import groupby
from typing import ClassVar, Dict, List

import colorama
from colorama import Fore

from ert.simulator.batch_simulator_context import Status
from everest.config import EverestConfig
from everest.detached import (
    OPT_PROGRESS_ID,
    SIM_PROGRESS_ID,
    ServerStatus,
    everserver_status,
    get_opt_status,
    start_monitor,
)
from everest.export import export
from everest.jobs import shell_commands
from everest.simulator import JOB_FAILURE, JOB_RUNNING, JOB_SUCCESS
from everest.strings import EVEREST

try:
    from progressbar import AdaptiveETA, Bar, Percentage, ProgressBar, Timer
except ImportError:
    ProgressBar = None  # type: ignore


def export_with_progress(config, export_ecl=True):
    logging.getLogger(EVEREST).info("Exporting results to csv ...")
    if ProgressBar is not None:
        widgets = [Percentage(), "  ", Bar(), "  ", Timer(), "  ", AdaptiveETA()]
        with ProgressBar(max_value=1, widgets=widgets) as bar:
            export_data = export(
                config=config, export_ecl=export_ecl, progress_callback=bar.update
            )
    else:
        export_data = export(config=config, export_ecl=export_ecl)

    return export_data


def export_to_csv(config: EverestConfig, data_frame=None, export_ecl=True):
    if data_frame is None:
        data_frame = export_with_progress(config, export_ecl)

    export_path = config.export_path
    output_folder = os.path.dirname(export_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data_frame.to_csv(export_path, sep=";", index=False)
    logging.getLogger(EVEREST).info("Data exported to {}".format(export_path))


def handle_keyboard_interrupt(signal, frame, options):
    print("\n" + "=" * 80)
    print(f"KeyboardInterrupt (ID: {signal}) has been caught. Program will exit...")
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


def _get_max_width(sequence):
    return max(len(item) for item in sequence)


def _format_list(values):
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
    status: Dict[str, List[int]] = field(
        default_factory=lambda: {
            JOB_RUNNING: [],  # contains running simulation numbers i.e [7,8,9]
            JOB_SUCCESS: [],  # contains successful simulation numbers i.e [0,1,3,4]
            JOB_FAILURE: [],  # contains failed simulation numbers i.e [5,6]
        }
    )
    STATUS_COLOR: ClassVar = {
        JOB_RUNNING: Fore.BLUE,
        JOB_SUCCESS: Fore.GREEN,
        JOB_FAILURE: Fore.RED,
    }

    def _status_string(self, max_widths: Dict[str, int]) -> str:
        string = []
        for state in [JOB_RUNNING, JOB_SUCCESS, JOB_FAILURE]:
            number_of_simulations = len(self.status[state])
            width = max_widths[state]
            color = self.STATUS_COLOR[state] if number_of_simulations else Fore.BLACK
            string.append(f"{color}{number_of_simulations:>{width}}{Fore.RESET}")
        return "/".join(string)

    def progress_str(self, max_widths: Dict[str, int]) -> str:
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

    def __init__(self, config, show_all_jobs):
        self._config = config
        self._show_all_jobs: bool = show_all_jobs
        self._clear_lines = 0
        self._batches_done = set()
        self._last_reported_batch = -1
        colorama.init(autoreset=True)

    def update(self, status):
        try:
            if OPT_PROGRESS_ID in status:
                opt_status = status[OPT_PROGRESS_ID]
                if opt_status and opt_status["cli_monitor_data"]:
                    msg, batch = self.get_opt_progress(opt_status)
                    if msg.strip():
                        # Clear the last reported batch of simulations if it
                        # should be after this optimization report:
                        if self._last_reported_batch > batch:
                            self._clear()
                        print(msg + "\n")
                        self._clear_lines = 0
            if SIM_PROGRESS_ID in status:
                sim_progress = status[SIM_PROGRESS_ID]
                sim_progress["status"] = Status(**sim_progress["status"])
                sim_progress["progress"] = self._filter_jobs(sim_progress["progress"])
                msg, batch = self.get_fm_progress(sim_progress)
                if msg.strip():
                    # Clear the previous report if it is still the same batch:
                    if batch == self._last_reported_batch:
                        self._clear()
                    print(msg)
                    self._clear_lines = len(msg.split("\n"))
                    self._last_reported_batch = batch
        except:
            logging.getLogger(EVEREST).debug(traceback.format_exc())

    def get_opt_progress(self, context_status):
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

    def _get_opt_progress_batch(self, cli_monitor_data, batch, idx):
        header = self._make_header(f"Optimization progress (Batch #{batch})")
        width = _get_max_width(cli_monitor_data["controls"][idx].keys())
        controls = self._join_one_newline_indent(
            f"{name:>{width}}: {value:{self.FLOAT_FMT}}"
            for name, value in cli_monitor_data["controls"][idx].items()
        )
        expected_objectives = cli_monitor_data["expected_objectives"]
        width = _get_max_width(expected_objectives.keys())
        objectives = self._join_one_newline_indent(
            f"{name:>{width}}: {value[idx]:{self.FLOAT_FMT}}"
            for name, value in expected_objectives.items()
        )
        objective_value = cli_monitor_data["objective_value"][idx]
        total_objective = (
            f"Total normalized objective: {objective_value:{self.FLOAT_FMT}}"
        )
        return self._join_two_newlines_indent(
            (header, controls, objectives, total_objective)
        )

    def get_fm_progress(self, context_status):
        batch_number = int(context_status["batch_number"])
        header = self._make_header(
            f"Running forward models (Batch #{batch_number})", Fore.BLUE
        )
        summary = self._get_progress_summary(context_status["status"])
        job_states = self._get_job_states(context_status["progress"])
        msg = self._join_two_newlines_indent((header, summary, job_states)) + "\n"
        return msg, batch_number

    @staticmethod
    def _get_progress_summary(status):
        colors = [
            Fore.BLACK,
            Fore.BLACK,
            Fore.BLUE if status[2] > 0 else Fore.BLACK,
            Fore.GREEN if status[3] > 0 else Fore.BLACK,
            Fore.RED if status[4] > 0 else Fore.BLACK,
        ]
        labels = ("Waiting", "Pending", "Running", "Complete", "FAILED")
        return " | ".join(
            f"{color}{key}: {value}{Fore.RESET}"
            for color, key, value in zip(colors, labels, status)
        )

    @classmethod
    def _get_job_states(cls, progress):
        print_lines = ""
        jobs_status = cls._get_jobs_status(progress)
        if jobs_status:
            max_widths = {
                state: _get_max_width(
                    str(len(item.status[state])) for item in jobs_status
                )
                for state in [JOB_RUNNING, JOB_SUCCESS, JOB_FAILURE]
            }
            width = _get_max_width([item.name for item in jobs_status])
            print_lines = cls._join_one_newline_indent(
                f"{item.name:>{width}}: {item.progress_str(max_widths)}{Fore.RESET}"
                for item in jobs_status
            )
        return print_lines

    @staticmethod
    def _get_jobs_status(progress):
        job_progress = {}
        for queue in progress:
            for job_idx, job in enumerate(queue):
                if job_idx not in job_progress:
                    job_progress[job_idx] = JobProgress(name=job["name"])
                simulation = int(job["simulation"])
                status = job["status"]
                if status in [JOB_RUNNING, JOB_SUCCESS, JOB_FAILURE]:
                    job_progress[job_idx].status[status].append(simulation)
        return job_progress.values()

    def _filter_jobs(self, progress):
        if not self._show_all_jobs:
            progress = [
                [job for job in progress_list if job["name"] not in shell_commands]
                for progress_list in progress
            ]
        return progress

    @classmethod
    def _join_one_newline_indent(cls, sequence):
        return ("\n" + " " * cls.INDENT).join(sequence)

    @classmethod
    def _join_two_newlines_indent(cls, sequence):
        return ("\n\n" + " " * cls.INDENT).join(sequence)

    @classmethod
    def _join_two_newlines(cls, sequence):
        return "\n\n".join(sequence)

    @classmethod
    def _make_header(cls, msg, color=Fore.BLACK):
        header = msg.center(len(msg) + 2).center(cls.WIDTH, "=")
        return f"{color}{header}{Fore.RESET}"

    def _clear(self):
        for _ in range(self._clear_lines):
            print(colorama.Cursor.UP(), end=colorama.ansi.clear_line())


def run_detached_monitor(config: EverestConfig, show_all_jobs: bool = False):
    monitor = _DetachedMonitor(config, show_all_jobs)
    start_monitor(config, callback=monitor.update)
    opt_status = get_opt_status(config.optimization_output_dir)
    if opt_status.get("cli_monitor_data"):
        msg, _ = monitor.get_opt_progress(opt_status)
        if msg.strip():
            print(f"{msg}\n")


def report_on_previous_run(config: EverestConfig):
    server_state = everserver_status(config)
    config_file = config.config_file
    if server_state["status"] == ServerStatus.failed:
        error_msg = server_state["message"]
        print(
            f"Optimization run failed, with error: {error_msg}\n"
            "To re-run optimization case use command:\n"
            f"`  everest run --new-run {config_file}`\n"
        )
    else:
        output_dir = config.output_dir
        opt_status = get_opt_status(config.optimization_output_dir)
        if opt_status.get("cli_monitor_data"):
            monitor = _DetachedMonitor(config, show_all_jobs=False)
            msg, _ = monitor.get_opt_progress(opt_status)
            print(msg + "\n")
        print(
            f"Optimization completed, results in {output_dir}\n"
            "\nTo re-run the optimization use command:\n"
            f"  `everest run --new-run {config_file}`\n"
            "To export the results use command:\n"
            f"  `everest export {config_file}`"
        )
