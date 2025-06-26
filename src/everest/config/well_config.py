import logging
from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
)

from ert.config import ConfigWarning
from everest.strings import EVEREST


class WellConfig(BaseModel):
    name: str = Field(description="The unique name of the well")
    drill_date: str | None = Field(
        None,
        description="""Ideal date to drill a well.

The interpretation of this is up to the forward model. The standard tooling will
consider this as the earliest possible drill date.
""",
    )
    drill_time: PositiveInt = Field(
        0,
        description="""specifies the time it takes
 to drill the well under consideration.""",
    )
    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("name")
    @classmethod
    def validate_no_dots_in_well_name(cls, well_name: str) -> str:
        if "." in well_name:
            raise ValueError("Well name cannot contain any dots (.)")

        return well_name

    @field_validator("drill_date")
    @classmethod
    def validate_drill_date_is_valid_date(cls, drill_date: str | None) -> str | None:
        if drill_date is None:
            return None

        try:
            parsed_date = datetime.strptime(drill_date, "%Y-%m-%d").date().isoformat()
        except ValueError as e:
            raise ValueError(
                f"malformed date: {drill_date}, expected format: YYYY-MM-DD"
            ) from e

        try:
            datetime.fromisoformat(drill_date)
        except ValueError:
            msg = (
                f"Deprecated date format: {drill_date}, "
                "please use ISO date format YYYY-MM-DD."
            )
            logging.getLogger(EVEREST).warning(msg)
            ConfigWarning.deprecation_warn(msg)
            return parsed_date

        return drill_date
