from typing import List, Literal, Optional

from pydantic import BaseModel, Field, NonNegativeInt, PositiveFloat, field_validator

from .sampler_config import SamplerConfig


class ControlVariableConfig(BaseModel, extra="forbid"):  # type: ignore
    name: str = Field(description="Control variable name")
    initial_guess: Optional[float] = Field(
        default=None,
        description="""
Starting value for the control variable, if given needs to be in the interval [min, max]
""",
    )
    control_type: Optional[Literal["real", "integer"]] = Field(
        default=None,
        description="""
The type of control. Set to "integer" for discrete optimization. This may be
ignored if the algorithm that is used does not support different control types.
""",
    )
    index: Optional[NonNegativeInt] = Field(
        default=None,
        description="""
Index should be given either for all of the variables or for none of them
""",
    )
    auto_scale: Optional[bool] = Field(
        default=None,
        description="""
Can be set to true to re-scale variable from the range
defined by [min, max] to the range defined by scaled_range (default [0, 1])
""",
    )
    scaled_range: Optional[List[float]] = Field(
        default=None,
        description="""
Can be used to set the range of the variable values
after scaling (default = [0, 1]).

This option has no effect if auto_scale is not set.
""",
    )
    min: Optional[float] = Field(
        default=None,
        description="""
Minimal value allowed for the variable

initial_guess is required to be greater than this value.
""",
    )
    max: Optional[float] = Field(
        default=None,
        description="""
    Max value allowed for the variable

    initial_guess is required to be less than this value.
    """,
    )
    perturbation_magnitude: Optional[PositiveFloat] = Field(
        default=None,
        description="""
Specifies the perturbation magnitude for this particular variable.
This feature adds flexibility to combine controls into more logical
of structures at the same time allowing the variable to contain time (
how long rate applies for) & value (the actual rate).

NOTE: In most cases this should not be configured, and the default value should be used.
""",
    )
    sampler: Optional[SamplerConfig] = Field(
        default=None, description="The backend used by Everest for sampling points"
    )

    @field_validator("name")
    @classmethod
    def validate_no_dots_in_name(cls, name: str) -> str:  # pylint: disable=E0213
        if "." in name:
            raise ValueError("Variable name can not contain any dots (.)")
        return name
