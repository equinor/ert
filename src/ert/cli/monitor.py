from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from queue import SimpleQueue
from typing import TextIO

import humanize
from tqdm import tqdm

from ert.ensemble_evaluator import (
    EndEvent,
    EnsembleSnapshot,
    FullSnapshotEvent,
    SnapshotUpdateEvent,
)
from ert.ensemble_evaluator import identifiers as ids
from ert.ensemble_evaluator.state import (
    COLOR_FAILED,
    COLOR_FINISHED,
    FORWARD_MODEL_STATE_FAILURE,
    REAL_STATE_TO_COLOR,
)
from ert.run_models.event import (
    RunModelDataEvent,
    RunModelErrorEvent,
    RunModelUpdateEndEvent,
    StatusEvents,
)

Color = tuple[int, int, int]


def _no_color(text: str, color: Color) -> str:
    """Alternate color method when no coloring is wanted"""
    return text


def ansi_color(text: str, color: Color) -> str:
    """Returns the text as a colorized text for an ansi terminal.

    https://en.wikipedia.org/wiki/ANSI_escape_code#24-bit
    """
    return "\x1b[38;2;{};{};{}m".format(*color) + text + "\x1b[0m"


class Monitor:
    """Class for tracking and outputting the progress of a simulation @model,
    where progress is defined as a combination of fields on @tracker.

    Progress is printed to @out. @color_always decides whether or not coloring
    always should take place, i.e. even if @out does not support it.
    """

    dot = "■ "

    def __init__(self, out: TextIO = sys.stdout, color_always: bool = False) -> None:
        self._out = out
        self._snapshots: dict[int, EnsembleSnapshot] = {}
        self._start_time: datetime | None = None
        self._colorize = ansi_color
        # If out is not (like) a tty, disable colors.
        if not out.isatty() and not color_always:
            self._colorize = _no_color

            # The dot adds no value without color, so remove it.
            self.dot = ""
        self.done = False

    def monitor(
        self,
        event_queue: SimpleQueue[StatusEvents],
        output_path: Path | None = None,
    ) -> EndEvent:
        self._start_time = datetime.now()
        while True:
            match event_queue.get():
                case FullSnapshotEvent(
                    snapshot=snapshot, iteration=iteration, progress=progress
                ):
                    if snapshot is not None:
                        self._snapshots[iteration] = snapshot
                    self._progress = progress
                case SnapshotUpdateEvent(snapshot=snapshot) as event:
                    if snapshot is not None:
                        self._snapshots[event.iteration].merge_snapshot(snapshot)
                    self._print_progress(event)
                case EndEvent() as event:
                    self._print_result(event.failed, event.msg)
                    self._print_step_errors()
                    return event
                case (
                    RunModelDataEvent()
                    | RunModelUpdateEndEvent()
                    | RunModelErrorEvent() as event
                ):
                    event.write_as_csv(output_path)

    def _print_step_errors(self) -> None:
        failed_steps: dict[str | None, int] = {}
        for snapshot in self._snapshots.values():
            for real in snapshot.reals.values():
                for step in real["fm_steps"].values():
                    if step.get(ids.STATUS) == FORWARD_MODEL_STATE_FAILURE:
                        err = step.get(ids.ERROR)
                        result = failed_steps.get(err, 0)
                        failed_steps[err] = result + 1
        for error, number_of_steps in failed_steps.items():
            print(f"{number_of_steps} steps failed due to the error: {error}")

    def _get_legends(self) -> str:
        statuses = ""
        latest_snapshot = self._snapshots[max(self._snapshots.keys())]
        total_count = len(latest_snapshot.reals)
        aggregate = latest_snapshot.aggregate_real_states()
        for state_ in reversed(REAL_STATE_TO_COLOR):
            count = aggregate[state_]
            countstring = f"{count}/{total_count}"
            out = (
                f"{self._colorize(self.dot, color=REAL_STATE_TO_COLOR[state_])}"
                f"{state_:10} {countstring:>10}"
            )
            statuses += f"    {out}\n"
        return statuses

    def _print_result(self, failed: bool, failed_message: str | None) -> None:
        if failed:
            msg = f"Experiment failed with the following error: {failed_message}"
            print(self._colorize(msg, color=COLOR_FAILED), file=self._out)
        else:
            print(
                self._colorize("Experiment completed.", color=COLOR_FINISHED),
                file=self._out,
            )

    def _print_progress(self, event: SnapshotUpdateEvent) -> None:
        current_iteration = min(event.total_iterations, event.iteration + 1)
        if self._start_time is not None:
            elapsed = datetime.now() - self._start_time
        else:
            elapsed = timedelta()

        nphase = f" {current_iteration}/{event.total_iterations}"

        bar_format = "   {desc} |{bar}| {percentage:3.0f}% {unit}"
        tqdm.write(f"    --> {event.iteration_label}", file=self._out)
        tqdm.write("\n", end="", file=self._out)
        with tqdm(total=100, ncols=100, bar_format=bar_format, file=self._out) as pbar:
            pbar.set_description_str(nphase, refresh=False)
            pbar.unit = f"Running time: {humanize.precisedelta(elapsed.seconds)}"
            pbar.update(event.progress * 100)
        tqdm.write("\n", end="", file=self._out)
        tqdm.write(self._get_legends(), file=self._out)
