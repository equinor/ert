from textwrap import dedent
from typing import Annotated, Any, Literal

from pydantic import (
    Discriminator,
    Field,
    model_validator,
)

from ert.base_model_context import BaseModelWithContextSupport


class ForwardModelResult(BaseModelWithContextSupport):
    file_name: str = Field(
        description=dedent(
            """
        The output file produced by the forward model.
        """
        )
    )


class SummaryResults(ForwardModelResult):
    type: Literal["summary"]
    keys: Literal["*"] | list[str] = Field(
        description=dedent(
            """
            The list of keys to include in the result.

            Defaults to '*' indicating all keys.
            """
        ),
        default="*",
    )


class GenDataResults(ForwardModelResult):
    type: Literal["gen_data"]


class ForwardModelStepConfig(BaseModelWithContextSupport):
    job: str = Field(
        description=dedent(
            """
            The forward model step to execute.

            A string that consists of the name of the job, followed by zero or
            more job options.
            """
        ),
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

    @model_validator(mode="before")
    @classmethod
    def check_discriminator(cls, values: dict[str, Any]) -> dict[str, Any]:
        results = values.get("results")
        if results is not None and "type" not in results:
            raise ValueError(
                "Missing required field 'type' in 'results'. This field is needed to "
                "determine the correct result schema (e.g., 'gen_data' or 'summary')."
                " Please include a 'type' key in the 'results' section."
            )
        return values
