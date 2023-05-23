from typing import List, Optional


class GenDataConfig:
    def __init__(
        self, key: str, input_file: str = "", report_steps: Optional[List[int]] = None
    ):
        self.name = key
        self._active_report_steps: List[int] = []
        self._observation_list: List[str] = []
        self.input_file = input_file
        if report_steps:
            self.add_report_steps(report_steps)
        else:
            self.add_report_step(0)

    def add_report_steps(self, steps: List[int]):
        for step in steps:
            self.add_report_step(step)

    def add_report_step(self, step: int):
        if not self.hasReportStep(step):
            self._active_report_steps.append(step)
            self._active_report_steps.sort()

    def update_observation_keys(self, observations: List[str]):
        self._observation_list = observations
        self._observation_list.sort()

    def get_observation_keys(self) -> List[str]:
        return self._observation_list

    def getKey(self) -> str:
        return self.name

    def __repr__(self):
        return (
            f"GenDataConfig(key={self.name}, "
            f"active_report_steps={self._active_report_steps}, "
            f"observation_keys={self._observation_list})"
        )

    def hasReportStep(self, report_step: int) -> bool:
        return report_step in self._active_report_steps

    def getNumReportStep(self) -> int:
        return len(self._active_report_steps)

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

        if self._observation_list != other._observation_list:
            return False

        return True
