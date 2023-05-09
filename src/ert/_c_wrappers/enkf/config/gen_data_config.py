from typing import List, Optional

from ert._c_wrappers.enkf.enums.ert_impl_type_enum import ErtImplType


class GenDataConfig:
    TYPE_NAME = "gen_data_config"

    def __init__(self, key: str, report_steps: Optional[List[int]] = None):
        self.name = key
        self._active_report_steps: List[int] = []
        if report_steps:
            self.add_report_steps(report_steps)

    def add_report_steps(self, steps: List[int]):
        for step in steps:
            self.add_report_step(step)

    def add_report_step(self, step: int):
        if not self.hasReportStep(step):
            self._active_report_steps.append(step)
            self._active_report_steps.sort()

    def getKey(self) -> str:
        return self.name

    def __repr__(self):
        return (
            f"GenDataConfig(key={self.name}, "
            f"active_report_steps={self._active_report_steps})"
        )

    def hasReportStep(self, report_step: int) -> bool:
        return report_step in self._active_report_steps

    def getNumReportStep(self) -> int:
        return len(self._active_report_steps)

    def getImplementationType(self) -> ErtImplType:
        return ErtImplType.GEN_DATA

    def getReportStep(self, index: int) -> int:
        return self._active_report_steps[index]

    def getReportSteps(self) -> List[int]:
        return [self.getReportStep(index) for index in range(self.getNumReportStep())]

    def __eq__(self, other) -> bool:
        if self.getKey() != other.getKey():
            return False

        if self.getNumReportStep() != other.getNumReportStep():
            return False

        if self.getReportSteps() != other.getReportSteps():
            return False

        return True
