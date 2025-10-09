from textwrap import dedent

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
)


class WellConfig(BaseModel):
    name: str = Field(
        description=dedent(
            """
        The unique name of the well.
        """
        )
    )
    drill_time: PositiveInt = Field(
        default=0,
        description=dedent(
            """
            Specifies the time it takes to drill the well.
            """,
        ),
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
