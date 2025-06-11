from typing import Annotated, Literal

from pydantic import (
    Discriminator,
    Field,
)

from ert.config.parsing import BaseModelWithContextSupport


class ForwardModelResult(BaseModelWithContextSupport):
    file_name: str = Field(description="Output file produced by the forward model")


class SummaryResults(ForwardModelResult):
    type: Literal["summary"] = "summary"
    keys: Literal["*"] | list[str] = Field(
        description="List of keys to include in the result. "
        "Defaults to '*' indicating all keys",
        default="*",
    )


class GenDataResults(ForwardModelResult):
    type: Literal["gen_data"] = "gen_data"


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
