from textwrap import dedent

from pydantic import BaseModel, Field


class CVaRConfig(BaseModel, extra="forbid"):
    percentile: float = Field(
        ge=0.0,
        le=1.0,
        description=dedent(
            """
            The percentile used for CVaR estimation.

            Sets the percentile of distribution of the objective over the
            realizations that is used to calculate the total objective.
            """
        ),
    )
