from ert_shared.status.entity import state
from ert_shared.ensemble_evaluator.entity import identifiers as ids
import math


def byte_with_unit(byte_count):
    suffixes = ["B", "kB", "MB", "GB", "TB", "PB"]
    power = float(10 ** 3)

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


def scale_intervals(reals):
    scaled_gen = _scale(reals, min_time=5, max_time=10)
    scaled_det = _scale(reals, min_time=10, max_time=20)
    return math.trunc(scaled_gen), math.trunc(scaled_det)


def _scale(nr_realizations, min_time=5, max_time=15, min_real=1, max_real=500):
    nr_realizations = min(max_real, nr_realizations)
    nr_realizations = max(min_real, nr_realizations)
    norm_real = _norm(min_real, max_real, nr_realizations)

    scaling_factor = _norm(_func(0), _func(1), _func(norm_real))
    return min_time + scaling_factor * (max_time - min_time)


def _norm(min_val, max_val, val):
    return (val - min_val) / (max_val - min_val)


def _func(val):
    return 1.0 * (1.0 + 500.0) ** val
