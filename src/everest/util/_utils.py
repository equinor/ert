from collections.abc import Sequence
from itertools import groupby


def format_list(values: Sequence[int]) -> str:
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
