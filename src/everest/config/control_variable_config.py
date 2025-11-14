from textwrap import dedent
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    model_validator,
)
from ropt.enums import VariableType

from ert.config import SamplerConfig

from .validation_utils import no_dots_in_string, valid_range


class _ControlVariable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, AfterValidator(no_dots_in_string)] = Field(
        description=dedent(
            """
            Variable name.

            The name of the variable must be unique within a control group,
            unless an `index` field is defined.
            """
        ),
    )
    control_type: Literal["real", "integer"] | None = Field(
        default=None,
        description=dedent(
            """
            Variable data type.

            If set, overrides the value of `control_type` of the control group.

            Set to `"integer"` for discrete optimization, or `"real"` for
            continuous optimization (default). This may be ignored if the
            optimization algorithm that is used does not support this.
            """
        ),
    )
    enabled: bool | None = Field(
        default=None,
        description=dedent(
            """
            Enable/disable the variable.

            If set to `False` the variable will be kept kept constant at the
            initial guess value during optimization.

            If set, overrides the value of `enabled` in the control group.
            """
        ),
    )
    scaled_range: Annotated[tuple[float, float] | None, AfterValidator(valid_range)] = (
        Field(
            default=None,
            description=dedent(
                """
            The target range of the variable scaling.

            Internally variables are scaled between their minimal and maximum
            values (`min`/`max` settings) to the range given by `target_range`
            (default = [0, 1]).

            Overrides the value of `scaled_range` in the control group.

            This option has no effect on discrete controls.
            """
            ),
        )
    )
    min: float | None = Field(
        default=None,
        description=dedent(
            """
            The minimum value of the variable kept during optimization.

            Overrides the value of `min` in the control group.

            The initial guess for this variable must respect this minimum value.
            """
        ),
    )
    max: float | None = Field(
        default=None,
        description=dedent(
            """
            The maximum value of the variable kept during optimization.

            Overrides the value of `max` in the control group.

            The initial guess for this variable must respect this maximum value.
            """
        ),
    )
    perturbation_type: Literal["absolute", "relative"] = Field(
        default="absolute",
        description=dedent(
            """
            The perturbation type for the control group.

            The `perturbation_type` keyword defines whether the perturbation
            magnitude (`perturbation_magnitude`) should be treated as an
            absolute value or as relative to the dynamic range of the controls.

            Overrides the value of `perturbation_type` in the control group.
            """
        ),
    )
    perturbation_magnitude: PositiveFloat | None = Field(
        default=None,
        description=dedent(
            """
            The perturbation magnitude for this variable.

            This controls the magnitude of perturbations (e.g. the standard
            deviation in case of a normal distribution) of controls, used to
            approximate the gradient.

            The interpretation of this field depends on the value of the
            `perturbation_type` field:

            - `absolute`: The given value is used as-is.
            - `relative`: The perturbation magnitude is calculated by
              multiplying the given by value by the difference between the `max`
              and `min` fields.

            Overrides the value of `perturbation_magnitude` in the control group.
            """
        ),
    )
    sampler: SamplerConfig | None = Field(
        default=None,
        description=dedent(
            """
            Sampler configuration for this variable.

            A sampler specifications is not required, if not provided, a normal
            distribution is used.

            If at least one control group or variable has a sampler
            specification, only the groups or variables with a sampler
            specification are perturbed. Controls/variables that do not have a
            sampler section will not be perturbed at all. If that is not
            desired, make sure to specify a sampler for each control group
            and/or variable (or none at all to use a normal distribution for
            each control).

            Within the sampler section, the `shared` keyword can be used to
            direct the sampler to use the same perturbations for each
            realization.

            Overrides the value of `sampler` in the control group.
            """
        ),
    )

    @property
    def ropt_control_type(self) -> VariableType | None:
        return VariableType[self.control_type.upper()] if self.control_type else None

    @model_validator(mode="before")
    @classmethod
    def check_for_autoscale_flag(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "auto_scale" in values:
            raise ValueError(
                "auto_scale is deprecated for everest controls, and is on by default."
            )
        return values


class ControlVariableConfig(_ControlVariable):
    model_config = ConfigDict(title="variable control")
    initial_guess: float | None = Field(
        default=None,
        description=dedent(
            """
            Initial guess for the variable.
            """
        ),
    )
    index: NonNegativeInt | None = Field(
        default=None,
        description=dedent(
            """
            A Non-negative integer value to distinguish variables with the same
            name within a control group.

            `index` should be given either for all of the variables, or for none
            of them.
            """
        ),
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
            description=dedent(
                """
                A List of initial values for the variables.

                If given, a series of variables with the same name are
                generated, with their `index` field set by enumeration, staring
                from 1. The initial value of these variables is taken from this
                list, whereas all other fields keep the same given value.
                """
            ),
        ),
    ]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and other.name == self.name

    @property
    def uniqueness(self) -> str:
        return "name"
