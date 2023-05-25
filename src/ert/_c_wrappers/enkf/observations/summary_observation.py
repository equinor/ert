from dataclasses import dataclass

from ert._c_wrappers.enkf.active_list import ActiveList
from ert._c_wrappers.enkf.enums import ActiveMode


@dataclass
class SummaryObservation:
    summary_key: str
    observation_key: str
    value: float
    std: float
    std_scaling: float = 1.0

    def getValue(self) -> float:
        return self.value

    def getStandardDeviation(self) -> float:
        return self.std

    def getStdScaling(self, index=0) -> float:
        return self.std_scaling

    def set_std_scaling(self, scaling_factor: float) -> None:
        self.std_scaling = scaling_factor

    def __len__(self):
        return 1

    def getSummaryKey(self) -> str:
        return self.summary_key

    def updateStdScaling(self, factor: float, active_list: ActiveList) -> None:
        if active_list.getMode() == ActiveMode.ALL_ACTIVE:
            self.std_scaling = factor
        elif active_list.getActiveSize() > 0:
            self.std_scaling = factor
