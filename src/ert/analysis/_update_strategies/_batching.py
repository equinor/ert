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
    if len(arr) == 0:
        return []

    sections = (len(arr) + batch_size - 1) // batch_size
    return np.array_split(arr, sections)


def calculate_localization_batch_size(num_params: int, num_obs: int) -> int:
    """Calculate a localization batch size that fits available memory."""
    if num_params == 0:
        return 0

    if num_obs == 0:
        return num_params

    available_memory_in_bytes = psutil.virtual_memory().available
    memory_safety_factor = 0.8
    bytes_in_float32 = 4
    batch_size = int(
        np.floor(
            (available_memory_in_bytes * memory_safety_factor)
            / (num_obs * bytes_in_float32)
        )
    )
    return max(1, min(batch_size, num_params))
