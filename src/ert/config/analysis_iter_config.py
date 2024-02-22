from typing import Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(alias_generator=lambda x: x.upper()))
class AnalysisIterConfig:
    iter_case: Optional[str] = None
    iter_count: int = 4
    iter_retry_count: int = 4
