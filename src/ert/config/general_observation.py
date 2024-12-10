from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class GenObservation:
    values: list[float]
    stds: list[float]
    indices: list[int]
    std_scaling: list[float]

    def __post_init__(self) -> None:
        for val in self.stds:
            if val <= 0:
                raise ValueError("Observation uncertainty must be strictly > 0")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GenObservation):
            return False
        return (
            np.array_equal(self.values, other.values)
            and np.array_equal(self.stds, other.stds)
            and np.array_equal(self.indices, other.indices)
            and np.array_equal(self.std_scaling, other.std_scaling)
        )

    def __len__(self) -> int:
        return len(self.values)
