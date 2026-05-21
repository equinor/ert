import numpy as np
import pytest

from ert.analysis._update_strategies._batching import (
    calculate_localization_batch_size,
    split_by_batch_size,
)


def test_that_batch_size_equals_num_params_when_no_observations() -> None:
    assert calculate_localization_batch_size(num_params=7, num_obs=0) == 7


def test_that_batch_size_is_clamped_to_one_when_available_memory_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class VirtualMemory:
        available = 1

    monkeypatch.setattr(
        "ert.analysis._update_strategies._batching.psutil.virtual_memory",
        VirtualMemory,
    )

    assert calculate_localization_batch_size(num_params=7, num_obs=1) == 1


def test_that_batch_size_is_zero_when_there_are_no_parameters() -> None:
    assert calculate_localization_batch_size(num_params=0, num_obs=7) == 0


def test_that_split_by_batch_size_keeps_batches_within_batch_size() -> None:
    batches = split_by_batch_size(np.arange(10), batch_size=6)

    assert [len(batch) for batch in batches] == [5, 5]
    assert all(len(batch) <= 6 for batch in batches)


def test_that_split_by_batch_size_handles_exact_division() -> None:
    batches = split_by_batch_size(np.arange(12), batch_size=6)

    assert [len(batch) for batch in batches] == [6, 6]


def test_that_split_by_batch_size_returns_one_batch_when_batch_size_is_larger() -> None:
    batches = split_by_batch_size(np.arange(5), batch_size=6)

    assert [batch.tolist() for batch in batches] == [[0, 1, 2, 3, 4]]


def test_that_split_by_batch_size_handles_empty_arrays() -> None:
    assert split_by_batch_size(np.array([], dtype=int), batch_size=6) == []
