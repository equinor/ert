import math


def byte_with_unit(byte_count):
    suffixes = ["B", "kB", "MB", "GB", "TB", "PB"]
    power = float(10**3)

    i = 0
    while byte_count >= power and i < len(suffixes) - 1:
        byte_count = byte_count / power
        i += 1

    return f"{byte_count:.2f} {suffixes[i]}"


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
