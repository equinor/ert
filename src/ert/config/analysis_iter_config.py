from typing import Optional

from pydantic import BaseModel


class AnalysisIterConfig(BaseModel):
    iter_case: Optional[str] = None
    iter_count: int = 4
    iter_retry_count: int = 4

    class Config:
        alias_generator = lambda x: x.upper()
