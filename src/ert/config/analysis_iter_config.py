from dataclasses import dataclass
from typing import Optional, no_type_check

from .parsing import ConfigDict, ConfigKeys


@dataclass
class AnalysisIterConfig:
    iter_case: Optional[str] = None
    iter_count: int = 4
    iter_retry_count: int = 4

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> "AnalysisIterConfig":
        iter_case: Optional[str] = config_dict.get(ConfigKeys.ITER_CASE)
        iter_count: int = config_dict.get(ConfigKeys.ITER_COUNT, 4)
        iter_retry_count: int = config_dict.get(ConfigKeys.ITER_RETRY_COUNT, 4)
        return cls(iter_case, iter_count, iter_retry_count)
