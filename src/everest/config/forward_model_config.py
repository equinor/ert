from typing import Annotated, Literal

from pydantic import (
    Discriminator,
    Field,
)

from ert.config.parsing import BaseModelWithContextSupport


class SummaryResults(BaseModelWithContextSupport):
    type: Literal["summary"] = "summary"
    file_name: str = Field(description="Output file produced by the forward model")
    keys: Literal["*"] | list[str] = Field(
        description="List of keys to include in the result. "
        "Defaults to '*' indicating all keys",
        default="*",
    )


class GenDataResults(BaseModelWithContextSupport):
    type: Literal["gen_data"] = "gen_data"
    file_name: str = Field(description="Output file produced by the forward model")


class ForwardModelStepConfig(BaseModelWithContextSupport):
    job: str = Field(
        description="Name of the forward model step",
    )
    results: (
        (
            Annotated[
                SummaryResults | GenDataResults,
                Discriminator("type"),
            ]
        )
        | None
    ) = Field(default=None)
