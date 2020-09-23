# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import threading
import time

from colors import color as ansi_color
from console_progressbar import ProgressBar

from ert_shared.tracker.events import EndEvent, GeneralEvent
from ert_shared.tracker.state import SimulationStateStatus
from ert_shared.tracker.utils import format_running_time


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
        for event in tracker.track():
            if isinstance(event, GeneralEvent):
                self._print_progress(event)
            if isinstance(event, EndEvent):
                self._print_result(event.failed, event.failed_msg)
                return

    def _get_legends(self, sim_states):
        legends = {}
        for state in sim_states:
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

    def _print_progress(self, event):
        """Print a progress based on the information on a GeneralEvent."""
        if event.indeterminate:
            # indeterminate, no progress to be shown
            return

        prefix = """
    --> {phase_name}

    {current_phase}/{target}""".format(
            phase_name=event.phase_name,
            current_phase=min(event.total_phases,
                              event.current_phase + 1),
            target=event.total_phases,
        )

        statuses = ""
        legends = self._get_legends(event.sim_states)
        for state in event.sim_states:
            statuses += "    {}\n".format(legends[state])

        suffix = """{runtime}

{statuses}""".format(statuses=statuses,
                     runtime=format_running_time(event.runtime))

        pb = ProgressBar(
            total=100, prefix=prefix, suffix=suffix, decimals=0,
            length=self.bar_length, fill=self.filled_bar_char,
            zfill=self.empty_bar_char, file=self._out
        )
        pb.print_progress_bar(event.progress * 100)
