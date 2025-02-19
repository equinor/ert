from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CVaRConfig(BaseModel):
    number_of_realizations: int | None = Field(
        default=None,
        description="""The number of realizations used for CVaR estimation.

Sets the number of realizations that is used to calculate the total objective.

This option is exclusive with the **percentile** option.
""",
    )
    percentile: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="""The percentile used for CVaR estimation.

Sets the percentile of distribution of the objective over the realizations that
is used to calculate the total objective.

This option is exclusive with the **number_of_realizations** option.

        """,
    )

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_mutex_nreals_percentile(cls, values: dict[str, Any]) -> dict[str, Any]:
        has_nreals = values.get("number_of_realizations") is not None
        has_percentile = values.get("percentile") is not None

        if not (has_nreals ^ has_percentile):
            raise ValueError(
                "Invalid CVaR section; Specify only one of the"
                " following: number_of_realizations, percentile"
            )

        return values
