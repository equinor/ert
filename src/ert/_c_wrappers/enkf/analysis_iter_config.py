from dataclasses import dataclass
from typing import Any, Dict, Optional

from ert._c_wrappers.enkf.config_keys import ConfigKeys


@dataclass
class AnalysisIterConfig:
    iter_case: Optional[str] = None
    iter_count: int = 4
    iter_retry_count: int = 4

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AnalysisIterConfig":
        iter_case = config_dict.get(ConfigKeys.ITER_CASE, None)
        iter_count = config_dict.get(ConfigKeys.ITER_COUNT, 4)
        iter_retry_count = config_dict.get(ConfigKeys.ITER_RETRY_COUNT, 4)
        return cls(iter_case, iter_count, iter_retry_count)
