import os.path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from ecl.util.util import IntVector

from ert._c_wrappers.enkf import ActiveList
from ert._c_wrappers.enkf.enums import ActiveMode

if TYPE_CHECKING:
    import numpy.typing as npt


class GenObservation:
    def __init__(
        self,
        scalar_value=None,
        obs_file: Optional[str] = None,
        data_index: Optional[str] = None,
    ):
        if scalar_value is None and obs_file is None:
            raise ValueError(
                "Exactly one the scalar_value and obs_file arguments must be present"
            )

        if scalar_value is not None and obs_file is not None:
            raise ValueError(
                "Exactly one the scalar_value and obs_file arguments must be present"
            )

        if obs_file is not None:
            values = np.loadtxt(obs_file, delimiter=None).ravel()
            if len(values) % 2 != 0:
                raise ValueError(
                    "Expected even number of values in GENERAL_OBSERVATION"
                )
            self.values = values[::2]
            self.stds = values[1::2]

        else:
            obs_value, obs_std = scalar_value
            self.values = np.array([obs_value])
            self.stds = np.array([obs_std])

        if data_index is not None:
            self.indices = []
            if os.path.isfile(data_index):
                self.indices = np.loadtxt(data_index, delimiter=None).ravel()
            else:
                self.indices = np.array(
                    IntVector.active_list(data_index), dtype=np.int32
                )
        else:
            self.indices = np.arange(len(self.values))
        self.std_scaling = [1.0] * len(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, obs_index: int) -> Tuple[float, float]:
        return (self.values[obs_index], self.stds[obs_index])

    def getValue(self, obs_index: int) -> float:
        return self.values[obs_index]

    def getStandardDeviation(self, obs_index: int) -> float:
        return self.stds[obs_index]

    def getStdScaling(self, obs_index: int) -> float:
        return self.std_scaling[obs_index]

    def updateStdScaling(self, factor: float, active_list: ActiveList) -> None:
        if active_list.getMode() == ActiveMode.ALL_ACTIVE:
            self.std_scaling = [factor] * len(self.values)
        else:
            for i in active_list.get_active_index_list():
                self.std_scaling[i] = factor

    def get_data_points(self) -> "npt.NDArray[np.double]":
        return self.values

    def get_std(self) -> "npt.NDArray[np.double]":
        return self.stds

    def getSize(self) -> int:
        return len(self)

    def getIndex(self, obs_index) -> int:
        return self.getDataIndex(obs_index)

    def getDataIndex(self, obs_index: int) -> int:
        return self.indices[obs_index]
