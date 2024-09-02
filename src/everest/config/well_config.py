from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from everest.strings import DATE_FORMAT


class WellConfig(BaseModel):
    name: str = Field(description="The unique name of the well")
    drill_date: Optional[str] = Field(
        None,
        description="""Ideal date to drill a well.

The interpretation of this is up to the forward model. The standard tooling will
consider this as the earliest possible drill date.
""",
    )
    drill_time: Optional[float] = Field(
        None,
        description="""specifies the time it takes
 to drill the well under consideration.""",
    )
    model_config = ConfigDict(
        extra="forbid",
    )

    @field_validator("drill_time")
    @classmethod
    def validate_positive_drill_time(cls, drill_time):  # pylint:disable=E0213
        if drill_time <= 0:
            raise ValueError("Drill time must be a positive number")

        return drill_time

    @field_validator("name")
    @classmethod
    def validate_no_dots_in_well_nane(cls, well_name):  # pylint:disable=E0213
        if "." in well_name:
            raise ValueError("Well name can not contain any dots (.)")

        return well_name

    @field_validator("drill_date")
    @classmethod
    def validate_drill_date_is_valid_date(cls, drill_date):  # pylint:disable=E0213
        try:
            if not isinstance(drill_date, str):
                raise ValueError("invalid type str expected")
            datetime.strptime(drill_date, DATE_FORMAT)
        except ValueError as e:
            raise ValueError(
                f"malformed date: {drill_date}, expected format: {DATE_FORMAT}"
            ) from e
