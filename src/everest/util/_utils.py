from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby
from pathlib import Path

from ert.storage import LocalExperiment, open_storage


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


def get_everest_experiment(storage_path: Path) -> LocalExperiment:
    """
    Creates everest storage from a storage path. Note: This
    requires there to be at least one initialized batch/ensemble
    for it to be possible to detect the experiment.
    """
    storage = open_storage(storage_path, mode="r")
    experiment = next(storage.experiments)
    assert isinstance(experiment, LocalExperiment)
    return experiment
