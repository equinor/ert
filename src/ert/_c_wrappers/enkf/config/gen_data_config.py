from dataclasses import dataclass
from typing import List

from sortedcontainers import SortedList

from ert._c_wrappers.enkf.config.response_config import ResponseConfig


@dataclass
class GenDataConfig(ResponseConfig):
    input_file: str = ""
    report_steps: SortedList = SortedList()

    def __post_init__(self):
        self.report_steps = (
            SortedList([0])
            if not self.report_steps
            else SortedList(set(self.report_steps))
        )

    def hasReportStep(self, report_step: int) -> bool:
        return report_step in self.report_steps

    def getReportSteps(self) -> List[int]:
        return self.report_steps
