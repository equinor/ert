from ert_shared.status.entity import state
from ert_shared.ensemble_evaluator.entity import identifiers as ids
import math


def byte_with_unit(byte_count):
    suffixes = ["B", "kB", "MB", "GB", "TB", "PB"]
    power = float(10**3)

    i = 0
    while byte_count >= power and i < len(suffixes) - 1:
        byte_count = byte_count / power
        i += 1

    return "{byte_count:.2f} {suffix}".format(byte_count=byte_count, suffix=suffixes[i])


def _calculate_progress(
    finished: bool,
    iter_: int,
    total_iter: int,
    done_reals: int,
    total_reals: int,
) -> float:
    if finished:
        return 1.0
    real_progress = float(done_reals) / total_reals
    return (iter_ + real_progress) / total_iter


def tracker_progress(tracker) -> float:
    if not tracker._iter_snapshot:
        return 0
    current_iter = max(list(tracker._iter_snapshot.keys()))
    done_reals = 0
    if current_iter in tracker._iter_snapshot:
        for real in tracker._iter_snapshot[current_iter].get_reals().values():
            if real.status in [
                state.REALIZATION_STATE_FINISHED,
                state.REALIZATION_STATE_FAILED,
            ]:
                done_reals += 1
    total_reals = len(tracker._iter_snapshot[current_iter].get_reals())
    return _calculate_progress(
        tracker.is_finished(),
        current_iter,
        tracker._model.phaseCount(),
        done_reals,
        total_reals,
    )


# This is not a case of not-invented-here, seems there is no good way of doing
# this in python's standard library.
def format_running_time(runtime: int) -> str:
    """Format runtime in seconds to a label ERT can use to indicate for how
    long an experiment has been running."""
    days = 0
    hours = 0
    minutes = 0
    seconds = math.trunc(runtime)

    if seconds >= 60:
        minutes, seconds = divmod(seconds, 60)

    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)

    if hours >= 24:
        days, hours = divmod(hours, 24)

    if days > 0:
        layout = "Running time: {d} days {h} hours {m} minutes {s} seconds"

    elif hours > 0:
        layout = "Running time: {h} hours {m} minutes {s} seconds"

    elif minutes > 0:
        layout = "Running time: {m} minutes {s} seconds"

    else:
        layout = "Running time: {s} seconds"

    return layout.format(d=days, h=hours, m=minutes, s=seconds)
