from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(eq=False)
class GenObservation:
    values: npt.NDArray[np.double]
    stds: npt.NDArray[np.double]
    indices: npt.NDArray[np.int32]
    std_scaling: npt.NDArray[np.double]

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
