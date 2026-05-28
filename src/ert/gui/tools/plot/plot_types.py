from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ObservationPlotLocations:
    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
