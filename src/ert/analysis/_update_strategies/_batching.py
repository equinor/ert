"""Batching helpers for update strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import psutil

if TYPE_CHECKING:
    import numpy.typing as npt


def split_by_batch_size(
    arr: npt.NDArray[np.int_], batch_size: int
) -> list[npt.NDArray[np.int_]]:
    """Split an array into sub-arrays of a specified batch size."""
    sections = 1 if batch_size > len(arr) else len(arr) // batch_size
    return np.array_split(arr, sections)


def calculate_localization_batch_size(num_params: int, num_obs: int) -> int:
    """Calculate a localization batch size that fits available memory."""
    if num_obs == 0:
        return num_params

    available_memory_in_bytes = psutil.virtual_memory().available
    memory_safety_factor = 0.8
    bytes_in_float32 = 4
    return min(
        int(
            np.floor(
                (available_memory_in_bytes * memory_safety_factor)
                / (num_obs * bytes_in_float32)
            )
        ),
        num_params,
    )
