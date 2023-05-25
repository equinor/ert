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

    def updateStdScaling(self, factor: float, active_list: ActiveList) -> None:
        if active_list.getMode() == ActiveMode.ALL_ACTIVE:
            self.std_scaling = factor
        elif active_list.getActiveSize() > 0:
            self.std_scaling = factor
