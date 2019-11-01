import math


def calculate_progress(
    phase, phase_count, finished, queue_running, queue_size, phase_has_run, done_count
):
    if finished:
        return 1.0
    if not queue_running and phase_has_run:
        # queue is not running, but it has run for this phase, so it's done
        return (phase + 1.0) / phase_count
    else:
        phase_progress = float(done_count) / queue_size
        return (phase + phase_progress) / phase_count


def format_running_time(runtime):
    """ @rtype: str """
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
    scaled_gen = _scale(reals, min_time=1, max_time=5)
    scaled_det = _scale(reals, min_time=1, max_time=15)
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
