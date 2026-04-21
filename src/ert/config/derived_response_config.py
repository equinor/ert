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
    def match_key(self) -> list[str]:
        """Matching columns for observations and responses. Along with
        'response_key' they create the key on which response data should match
        observation data."""

    @property
    def index_key(self) -> list[str]:
        """Identification columns for observations."""
        return self.match_key

    def index_column_expr(self) -> pl.Expr:
        """Polars expression building the textual "index" column.

        Concatenates `index_key` columns with ", ". Null components are
        rendered as the literal "None" so positional meaning is preserved.
        """
        return pl.concat_str(
            [pl.col(c).cast(pl.String).fill_null("None") for c in self.index_key],
            separator=", ",
        )

    def display_column(self, value: Any, column_name: str) -> str:
        return str(value)

    @property
    def filter_on(self) -> dict[str, dict[str, list[int]]] | None:
        """Optional filters for this response."""
        return None
