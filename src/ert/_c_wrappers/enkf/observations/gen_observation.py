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

    def __eq__(self, other):
        return (
            np.array_equal(self.values, other.values)
            and np.array_equal(self.stds, other.stds)
            and np.array_equal(self.indices, other.indices)
            and np.array_equal(self.std_scaling, other.std_scaling)
        )

    def __len__(self):
        return len(self.values)
