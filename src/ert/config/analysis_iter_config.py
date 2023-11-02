from dataclasses import dataclass
from typing import Optional

from typing_extensions import Self

from ._config_values import DEFAULT_ITER_COUNT, DEFAULT_RETRY_COUNT, ErtConfigValues


@dataclass
class AnalysisIterConfig:
    iter_case: Optional[str] = None
    iter_count: int = DEFAULT_ITER_COUNT
    iter_retry_count: int = DEFAULT_RETRY_COUNT

    @classmethod
    def from_values(cls, config_values: ErtConfigValues) -> Self:
        return cls(
            config_values.iter_case,
            config_values.iter_count,
            config_values.iter_retry_count,
        )
