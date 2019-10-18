# -*- coding: utf-8 -*-
from __future__ import print_function

import sys

from colors import color as ansi_color
from console_progressbar import ProgressBar

from ert_shared.models import SimulationStateStatus, SimulationsTracker


def _ansi_color(*args, **kwargs):
    # This wraps ansi_color such that when _ansi_color is bound to Monitor,
    # all arguments are passed to ansi_color except the instance (self).
    return ansi_color(*args[1:], **kwargs)


class Monitor(object):
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

    def monitor(self, tracker):
        for update in tracker.track():
            self._print_progress(update)

        self._print_result(tracker.run_failed, tracker.failed_message)

    def _get_legends(self, tracker):
        legends = {}
        for state in tracker.getStates():
            legends[state] = "{}{:10} {:>10}".format(
                self._colorize(self.dot, fg=state.color), state.name,
                "{}/{}".format(state.count, state.total_count)
            )
        return legends

    def _print_result(self, failed, failed_message):
        if failed:
            msg = "Simulations failed with the following error: {}".format(
                failed_message)
            print(self._colorize(msg, fg=SimulationStateStatus.COLOR_FAILED),
                  file=self._out)
        else:
            print(self._colorize("Simulations completed.",
                                 fg=SimulationStateStatus.COLOR_FINISHED),
                  file=self._out)

    def _print_progress(self, tracker):
        """Print a progress based on the information on a SimulationTracker
        instance @tracker."""
        if tracker.queue_size == 0:
            # queue_size is 0, so no progress can be displayed
            return

        prefix = """
    --> {phase_name}

    {current_phase}/{target}""".format(
            phase_name=tracker.iteration_name,
            current_phase=min(tracker.total_iterations,
                              tracker.current_iteration + 1),
            target=tracker.total_iterations,
        )

        statuses = ""
        done = 0
        legends = self._get_legends(tracker)
        for state in tracker.getStates():
            statuses += "    {}\n".format(legends[state])
            if state.name == "Finished":
                done = state.count

        suffix = """{runtime}

{statuses}""".format(statuses=statuses,
                     runtime=SimulationsTracker.format_running_time(
                         tracker.runtime),)

        pb = ProgressBar(
            total=tracker.queue_size, prefix=prefix, suffix=suffix, decimals=0,
            length=self.bar_length, fill=self.filled_bar_char,
            zfill=self.empty_bar_char, file=self._out
        )
        pb.print_progress_bar(done)
