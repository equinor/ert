from abc import abstractmethod
from typing import Any

import polars as pl
from pydantic import BaseModel, Field


class DerivedResponseConfig(BaseModel):
    type: str
    keys: list[str] = Field(default_factory=list)
    has_finalized_keys: bool = False

    @abstractmethod
    def derive_from_storage(self, iter_: int, real: int, ensemble: Any) -> pl.DataFrame:
        """Derives response DataFrame from existing files in storage"""

    @property
    @abstractmethod
    def primary_key(self) -> list[str]:
        """Primary key of this response data.
        For example 'time' for summary and ['index','report_step'] for gen data"""

    def display_column(self, value: Any, column_name: str) -> str:
        return str(value)
