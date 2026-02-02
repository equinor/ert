from abc import abstractmethod
from enum import StrEnum
from typing import Any, Self

import polars as pl
from pydantic import BaseModel, Field

from .parsing import ConfigDict


class InvalidResponseFile(Exception):
    """
    Raised when an input file of the ResponseConfig has
    the incorrect format.
    """


class ResponseType(StrEnum):
    summary = "summary"
    gen_data = "gen_data"
    rft = "rft"
    breakthrough = "breakthrough"
    everest_constraints = "everest_constraints"
    everest_objectives = "everest_objectives"


response_primary_keys = {
    ResponseType.gen_data: ["report_step", "index"],
    ResponseType.summary: ["time"],
    ResponseType.rft: ["east", "north", "tvd"],
    ResponseType.breakthrough: ["threshold"],
    ResponseType.everest_constraints: [],
    ResponseType.everest_objectives: [],
}


class ResponseConfig(BaseModel):
    type: ResponseType
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
    def primary_key(self) -> list[str]:
        return response_primary_keys[self.type]

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
