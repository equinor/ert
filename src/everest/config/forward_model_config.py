from typing import Literal

from pydantic import (
    Field,
)

from ert.config.parsing import BaseModelWithContextSupport


class ForwardModelStepResultsConfig(BaseModelWithContextSupport):
    file_name: str = Field(description="Output file produced by the forward model")
    type: Literal["gen_data", "summary"] = Field(
        description="Type of the result, either 'gen_data' or 'summary'"
    )
    keys: Literal["*"] | list[str] = Field(
        description="List of keys to include in the result. "
        "Defaults to '*' indicating all keys",
        default="*",
    )


class ForwardModelStepConfig(BaseModelWithContextSupport):
    job: str = Field(
        description="Name of the forward model step",
    )
    results: ForwardModelStepResultsConfig | None = Field(
        default=None, description="Result file produced by the forward model"
    )
