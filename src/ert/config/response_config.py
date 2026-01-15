from abc import abstractmethod
from typing import Any, Self

import polars as pl
from pydantic import BaseModel, Field

from .parsing import ConfigDict


class InvalidResponseFile(Exception):
    """
    Raised when an input file of the ResponseConfig has
    the incorrect format.
    """


class ResponseConfig(BaseModel):
    type: str
    input_files: list[str] = Field(default_factory=list)
    keys: list[str] = Field(default_factory=list)
    has_finalized_keys: bool = False

    @abstractmethod
    def read_from_file(self, run_path: str, iens: int, iter_: int) -> pl.DataFrame:
        """Reads the data for the response from run_path.

        Raises:
            FileNotFoundError: when one of the input_files for the
                response is missing.
            InvalidResponseFile: when one of the input_files is
                invalid
        """

    @property
    @abstractmethod
    def expected_input_files(self) -> list[str]:
        """Returns a list of filenames expected to be produced by the forward model"""

    @property
    @abstractmethod
    def primary_key(self) -> list[str]:
        """Primary key of this response data.
        For example 'time' for summary and ['index','report_step'] for gen data"""

    @classmethod
    @abstractmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> Self | None:
        """Creates a config, given an ert config dict.
        A response config may depend on several config kws, such as REFCASE
        for summary."""

    @classmethod
    def display_column(cls, value: Any, column_name: str) -> str:
        """Formats a value to a user-friendly displayable format."""
        return str(value)

    @property
    def filter_on(self) -> dict[str, dict[str, list[int]]] | None:
        """Optional filters for this response."""
        return None
