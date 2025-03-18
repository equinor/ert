from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
)
from ropt.enums import VariableType

from .sampler_config import SamplerConfig
from .validation_utils import no_dots_in_string, valid_range


class _ControlVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, AfterValidator(no_dots_in_string)] = Field(
        description="Control variable name"
    )
    control_type: Literal["real", "integer"] | None = Field(
        default=None,
        description="""
The type of control. Set to "integer" for discrete optimization. This may be
ignored if the algorithm that is used does not support different control types.
""",
    )
    enabled: bool | None = Field(
        default=None,
        description="""
If `True`, the variable will be optimized, otherwise it will be fixed to the
initial value.
""",
    )
    auto_scale: bool | None = Field(
        default=None,
        description="""
Can be set to true to re-scale variable from the range
defined by [min, max] to the range defined by scaled_range (default [0, 1]).
""",
    )
    scaled_range: Annotated[tuple[float, float] | None, AfterValidator(valid_range)] = (
        Field(
            default=None,
            description="""
Can be used to set the range of the variable values
after scaling (default = [0, 1]).

This option has no effect on discrete controls.
""",
        )
    )
    min: float | None = Field(
        default=None,
        description="""
Minimal value allowed for the variable

initial_guess is required to be greater than this value.
""",
    )
    max: float | None = Field(
        default=None,
        description="""
    Max value allowed for the variable

    initial_guess is required to be less than this value.
    """,
    )
    perturbation_magnitude: PositiveFloat | None = Field(
        default=None,
        description="""
Specifies the perturbation magnitude for this particular variable.
This feature adds flexibility to combine controls into more logical
of structures at the same time allowing the variable to contain time (
how long rate applies for) & value (the actual rate).

NOTE: In most cases this should not be configured, and the default value should be used.
""",
    )
    sampler: SamplerConfig | None = Field(
        default=None, description="The backend used by Everest for sampling points"
    )

    @property
    def ropt_control_type(self) -> VariableType | None:
        return VariableType[self.control_type.upper()] if self.control_type else None


class ControlVariableConfig(_ControlVariable):
    model_config = ConfigDict(title="variable control")
    initial_guess: float | None = Field(
        default=None,
        description="""
Starting value for the control variable, if given needs to be in the interval [min, max]
""",
    )
    index: NonNegativeInt | None = Field(
        default=None,
        description="""
Index should be given either for all of the variables or for none of them
""",
    )

    def __hash__(self) -> int:
        return hash(self.name) + hash(self.index)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and other.name == self.name
            and other.index == self.index
        )

    @property
    def uniqueness(self) -> str:
        return "name-index"


class ControlVariableGuessListConfig(_ControlVariable):
    initial_guess: Annotated[
        list[float],
        Field(
            default=None,
            description="List of Starting values for the control variable",
        ),
    ]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and other.name == self.name

    @property
    def uniqueness(self) -> str:
        return "name"
