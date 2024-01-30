from typing import List, Literal, Optional

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
)
from typing_extensions import Annotated

from everest.config.validation_utils import no_dots_in_string

from .sampler_config import SamplerConfig


class _ControlVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, AfterValidator(no_dots_in_string)] = Field(
        description="Control variable name"
    )
    control_type: Optional[Literal["real", "integer"]] = Field(
        default=None,
        description="""
The type of control. Set to "integer" for discrete optimization. This may be
ignored if the algorithm that is used does not support different control types.
""",
    )
    enabled: Optional[bool] = Field(
        default=None,
        description="""
If `True`, the variable will be optimized, otherwise it will be fixed to the
initial value.
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


class ControlVariableConfig(_ControlVariable):
    initial_guess: Optional[float] = Field(
        default=None,
        description="""
Starting value for the control variable, if given needs to be in the interval [min, max]
""",
    )
    index: Optional[NonNegativeInt] = Field(
        default=None,
        description="""
Index should be given either for all of the variables or for none of them
""",
    )


class ControlVariableGuessListConfig(_ControlVariable):
    initial_guess: List[float] = Field(
        default=None,
        description="List of Starting values for the control variable",
    )
