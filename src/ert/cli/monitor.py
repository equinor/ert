# -*- coding: utf-8 -*-
import sys
from datetime import datetime
from typing import Iterator, Union

from colors import color as ansi_color
from tqdm import tqdm

from ert.ensemble_evaluator import EndEvent, FullSnapshotEvent, SnapshotUpdateEvent
from ert.ensemble_evaluator.state import (
    ALL_REALIZATION_STATES,
    COLOR_FAILED,
    COLOR_FINISHED,
    JOB_STATE_FAILURE,
    REAL_STATE_TO_COLOR,
)
from ert.shared.status.utils import format_running_time


def _ansi_color(*args, **kwargs):
    # This wraps ansi_color such that when _ansi_color is bound to Monitor,
    # all arguments are passed to ansi_color except the instance (self).
    return ansi_color(*args[1:], **kwargs)


class Monitor:
    """Class for tracking and outputting the progress of a simulation @model,
    where progress is defined as a combination of fields on @tracker.

    Progress is printed to @out. @color_always decides whether or not coloring
    always should take place, i.e. even if @out does not support it.
    """

    dot = "■ "
    empty_bar_char = " "
    filled_bar_char = "█"
    bar_length = 30

    _colorize = _ansi_color

    def __init__(self, out=sys.stdout, color_always=False):
        self._out = out
        self._snapshots = {}
        self._start_time = None
        # If out is not (like) a tty, disable colors.
        if not out.isatty() and not color_always:
            self._colorize = self._no_color

            # The dot adds no value without color, so remove it.
            self.dot = ""

    def _no_color(self, *args, **kwargs):
        """Alternate color method when no coloring is wanted. Conforms to the
        signature of ansi_color.color, wherein the first positional argument
        is the string to be (un-)colored."""
        return args[0]

    def monitor(
        self,
        events: Iterator[Union[FullSnapshotEvent, SnapshotUpdateEvent, EndEvent]],
    ):
        self._start_time = datetime.now()
        for event in events:
            if isinstance(event, FullSnapshotEvent):
                if event.snapshot is not None:
                    self._snapshots[event.iteration] = event.snapshot
                self._progress = event.progress
            elif isinstance(event, SnapshotUpdateEvent):
                if event.partial_snapshot is not None:
                    self._snapshots[event.iteration].merge_event(event.partial_snapshot)
                self._print_progress(event)
            if isinstance(event, EndEvent):
                self._print_result(event.failed, event.failed_msg)
                self._print_job_errors()
                return

    def _print_job_errors(self):
        failed_jobs = {}
        for snapshot_id, snapshot in self._snapshots.items():
            for real_id, real in snapshot.reals.items():
                for step_id, step in real.steps.items():
                    for job_id, job in step.jobs.items():
                        if job.status == JOB_STATE_FAILURE:
                            result = failed_jobs.get(job.error, 0)
                            failed_jobs[job.error] = result + 1
        for error, number_of_jobs in failed_jobs.items():
            print(f"{number_of_jobs} jobs failed due to the error: {error}")

    def _get_legends(self) -> str:
        statuses = ""
        latest_snapshot = self._snapshots[max(self._snapshots.keys())]
        total_count = len(latest_snapshot.reals)
        aggregate = latest_snapshot.aggregate_real_states()
        for state_ in ALL_REALIZATION_STATES:
            count = 0
            if state_ in aggregate:
                count = aggregate[state_]
            _countstring = f"{count}/{total_count}"
            out = (
                f"{self._colorize(self.dot, fg=REAL_STATE_TO_COLOR[state_])}"
                f"{state_:10} {_countstring:>10}"
            )
            statuses += f"    {out}\n"
        return statuses

    def _print_result(self, failed, failed_message):
        if failed:
            msg = f"Simulations failed with the following error: {failed_message}"
            print(self._colorize(msg, fg=COLOR_FAILED), file=self._out)
        else:
            print(
                self._colorize("Simulations completed.", fg=COLOR_FINISHED),
                file=self._out,
            )

    def _print_progress(self, event):
        """Print a progress based on the information on a GeneralEvent."""
        if event.indeterminate:
            # indeterminate, no progress to be shown
            return

        current_phase = min(event.total_phases, event.current_phase + 1)
        elapsed = datetime.now() - self._start_time

        nphase = f" {current_phase}/{event.total_phases}"

        bar_format = "   {desc} |{bar}| {percentage:3.0f}% {unit}"
        tqdm.write(f"    --> {event.phase_name}", file=self._out)
        tqdm.write("\n", end="", file=self._out)
        with tqdm(total=100, ncols=100, bar_format=bar_format, file=self._out) as pbar:
            pbar.set_description_str(nphase, refresh=False)
            pbar.unit = f"{format_running_time(elapsed.seconds)}"
            pbar.update(event.progress * 100)
        tqdm.write("\n", end="", file=self._out)
        tqdm.write(self._get_legends(), file=self._out)
