import warnings
from abc import abstractmethod
from typing import Any, Literal, Self

import polars as pl
from pydantic import BaseModel, Field

from ert.utils import assert_schema
from ert.warnings import PostExperimentWarning

from .parsing import ConfigDict

_RESPONSE_WARNING_LIMIT = 5


def _warn_about_missing_responses(
    missing_items: list[str],
    item_label: Literal["key(s)", "well(s) at time(s)"],
    filename: str,
) -> None:
    if not missing_items:
        return

    num_excess = len(missing_items) - _RESPONSE_WARNING_LIMIT

    warning = (
        f"Could not find responses for {item_label} in '{filename}':\n"
        + "\n".join(missing_items[:_RESPONSE_WARNING_LIMIT])
        + (f"\n... and {num_excess} other missing responses" if num_excess > 0 else "")
    )
    warnings.warn(
        warning,
        PostExperimentWarning,
        stacklevel=1,
    )


class InvalidResponseFile(Exception):
    """
    Raised when an input file of the ResponseConfig has
    the incorrect format.
    """


class ResponseConfig(BaseModel, extra="forbid"):
    """Represents an abstract response configuration in the ERT config.

    Some attributes are the same for all children classes, yet moving them to the base
    class causes troubles with pydantic serialization. When fields order changes, it
    changes fields serialization, breaking many tests. So instead getter and setter-like
    abstract methods are used as of now.
    """

    type: str

    @property
    @abstractmethod
    def match_key(self) -> list[str]:
        """Matching columns for observations and responses. Along with
        'response_key' they create the key on which response data should match
        observation data. For example 'time' for summary and ['report_step','index'] for
        gen data.
        """

    def match_key_dict_expr(self) -> pl.Expr:
        """Polars expression for representing match key-values as a label=value string.

        Concatenates `match_key` columns with ", ". Null components are rendered as the
        literal "None" so positional meaning is preserved.
        """
        return pl.concat_str(
            [
                pl.concat_str(
                    [pl.lit(f"{col}="), pl.col(col).cast(pl.String).fill_null("None")],
                    separator="",
                )
                for col in self.match_key
            ],
            separator=", ",
        )

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

    @classmethod
    def display_column(cls, value: Any, column_name: str) -> str:
        """Formats a value to a user-friendly displayable format."""
        return str(value)

    @property
    def filter_on(self) -> dict[str, dict[str, list[int]]] | None:
        """Optional filters for this response."""
        return None

    @abstractmethod
    def response_keys(self) -> list[str]:
        """Identifiers for response datasets this config implicitly produces.

        A response config may produce multiple independent datasets from a single
        simulation run — for example, one per well, one per summary key, or one per
        output file. Each such dataset is assigned a key. What constitutes a "dataset"
        is a design decision: it defines the finest granularity at which responses can
        be independently loaded or be matched against observations "dataset" with the
        same key.
        """

    @abstractmethod
    def are_keys_finalized(self) -> bool:
        """
        True if keys are finalized, False otherwise (for example, keys were declared
        with wildcard and have not been resolved yet).
        """

    @abstractmethod
    def finalize_keys(self, keys: list[str]) -> None:
        """
        Finalizes the keys for this response config. This is called when the keys are
        resolved (for example, when wildcards are expanded).
        """

    @abstractmethod
    def is_derived(self) -> bool:
        """
        Indicates whether response is derived from other data (not directly
        produced by forward model).
        """

    @staticmethod
    def _assert_schema(df: pl.DataFrame, schema: dict[str, Any]) -> pl.DataFrame:
        return assert_schema(df, schema)


class SimulationResponseConfig(ResponseConfig):
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

    @classmethod
    @abstractmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> Self | None:
        """Creates a config, given an ert config dict.
        A response config may depend on several config kws, such as REFCASE
        for summary.
        """

    def response_keys(self) -> list[str]:
        return self.keys

    def are_keys_finalized(self) -> bool:
        return self.has_finalized_keys

    def finalize_keys(self, keys: list[str]) -> None:
        self.keys = keys
        self.has_finalized_keys = True

    def is_derived(self) -> bool:
        return False


class DerivedResponseConfig(ResponseConfig):
    keys: list[str] = Field(default_factory=list)
    has_finalized_keys: bool = False

    @abstractmethod
    def derive_from_storage(self, iter_: int, real: int, ensemble: Any) -> pl.DataFrame:
        """Derives response DataFrame from existing files in storage"""

    def response_keys(self) -> list[str]:
        return self.keys

    def are_keys_finalized(self) -> bool:
        return self.has_finalized_keys

    def finalize_keys(self, keys: list[str]) -> None:
        self.keys = keys
        self.has_finalized_keys = True

    def is_derived(self) -> bool:
        return True
