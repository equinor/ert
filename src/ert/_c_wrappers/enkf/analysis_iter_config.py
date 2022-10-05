from dataclasses import dataclass
from typing import Any, Dict

from ert._c_wrappers.enkf.config_keys import ConfigKeys


@dataclass
class AnalysisIterConfig:
    iter_case: str = "ITERATED_ENSEMBLE_SMOOTHER%d"
    iter_count: int = 4
    iter_retry_count: int = 4
    case_format_set: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AnalysisIterConfig":
        iter_case = config_dict.get(
            ConfigKeys.ITER_CASE, "ITERATED_ENSEMBLE_SMOOTHER%d"
        )
        iter_count = config_dict.get(ConfigKeys.ITER_COUNT, 4)
        iter_retry_count = config_dict.get(ConfigKeys.ITER_RETRY_COUNT, 4)
        return cls(iter_case, iter_count, iter_retry_count)

    def get_num_iterations(self) -> int:
        return self.iter_count

    def __len__(self) -> int:
        return self.iter_count

    def set_num_iterations(self, num_iterations):
        self.iter_count = num_iterations

    def get_num_retries(self) -> int:
        return self.iter_retry_count

    def case_format(self) -> str:
        return self.iter_case

    def case_format_is_set(self) -> bool:
        return self.case_format_set

    def set_case_format(self, case_fmt):
        self.iter_case = case_fmt
        self.case_format_set = True

    def _short_case_fmt(self, maxlen=10):
        if len(self.iter_case) <= maxlen:
            return self.iter_case
        return self.iter_case[: maxlen - 2] + ".."

    def __repr__(self):
        cfs = f"format = {self.case_format_is_set()}"
        fmt = self._short_case_fmt()
        its = self.iter_count
        rets = self.iter_retry_count
        ret = "AnalysisIterConfig(iterations = %d, retries = %d, fmt = %s, %s)"
        return ret % (its, rets, fmt, cfs)

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        if self.case_format() != other.case_format():
            return False

        if self.get_num_iterations() != other.get_num_iterations():
            return False

        if self.get_num_retries() != other.get_num_retries():
            return False

        return True
