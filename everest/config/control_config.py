from itertools import chain
from typing import List, Literal, Optional

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated, Self

from .control_variable_config import ControlVariableConfig
from .sampler_config import SamplerConfig
from .validation_utils import no_dots_in_string

VARIABLE_ERROR_MESSAGE = (
    "Variable {name} must define {variable_type} value either"
    " at control level or variable level"
)


def firgure_a_better_name(
    name: str,
    _min: Optional[float],
    _max: Optional[float],
    initial_guess: Optional[float],
) -> List[str]:
    error = []
    if _min is None:
        error.append(VARIABLE_ERROR_MESSAGE.format(name=name, variable_type="min"))
    if _max is None:
        error.append(VARIABLE_ERROR_MESSAGE.format(name=name, variable_type="max"))
    if initial_guess is None:
        error.append(
            VARIABLE_ERROR_MESSAGE.format(name=name, variable_type="initial_guess")
        )

    if (
        _min is not None
        and _max is not None
        and initial_guess is not None
        and not _min <= initial_guess <= _max
    ):
        error.append(f"Variable {name} must respect min <= initial_guess <= max")
    return error


class ControlConfig(BaseModel):
    name: Annotated[str, AfterValidator(no_dots_in_string)] = Field(
        description="Control name"
    )
    type: Literal["well_control", "generic_control"] = Field(
        description="""
Only two allowed control types are accepted

* **well_control**: Standard built-in Everest control type designed for field\
 optimization

* **generic_control**: Enables the user to define controls types to be employed for\
 customized optimization jobs.
"""
    )
    variables: List[ControlVariableConfig] = Field(
        description="List of control variables"
    )
    initial_guess: Optional[float] = Field(
        default=None,
        description="""
Initial guess for the control group all control variables with initial_guess not
defined will be assigned this value. Individual initial_guess values in the control
variables will overwrite this value.
""",
    )
    control_type: Optional[Literal["real", "integer"]] = Field(
        default=None,
        description="""
The type of the controls for the control group. Individual control types in the
control variables will override this value. Set to "integer" for discrete
optimization. This may be ignored if the algorithm that is used does not support
different control types.
""",
    )
    enabled: Optional[bool] = Field(
        default=True,
        description="""
If `True`, all variables in this control group will be optimized. If set to `False`
the value of the variables will remain fixed.
""",
    )
    auto_scale: Optional[bool] = Field(
        default=None,
        description="""
Can be set to true to re-scale controls from the range
defined by [min, max] to the range defined by
scaled_range (default [0, 1]).
        """,
    )
    min: Optional[float] = Field(
        default=None,
        description="""
Defines left-side value in the control group range [min, max].
This value will be overwritten by the control variable min value if given.

The initial guess for both the group and the individual variables needs to be contained
in the resulting [min, max] range
""",
    )
    max: Optional[float] = Field(
        default=None,
        description="""
Defines right-side value in the control group range [min, max].
This value will be overwritten by the control variable max value if given.

The initial guess for both the group and the individual variables needs to be contained
in the resulting [min, max] range
""",
    )
    perturbation_type: Optional[Literal["absolute", "relative"]] = Field(
        default=None,
        description="""
Example: absolute or relative
Specifies the perturbation type for a set of controls of a certain type.  The
 perturbation type keyword defines whether the perturbation magnitude
 (perturbation_magnitude) should be considered as an absolute value or relative
 to the dynamic range of the controls.

NOTE: currently the dynamic range is computed with respect to all controls, so
 defining relative perturbation type for control types with different dynamic
 ranges might have unintended effects.
        """,
    )
    perturbation_magnitude: Optional[float] = Field(
        default=None,
        description="""
Specifies the perturbation magnitude for a set of controls of a certain type.

This controls the size of perturbations (standard deviation of a
normal distribution) of controls used to approximate the gradient.
The value depends on the type of control and magnitude of the variables.
For continuous controls smaller values should give a better gradient,
whilst for more discrete controls larger values should give a better
result. However, this is a balance as too large or too small
of values also cause issues.

NOTE: In most cases this should not be configured, and the default value should be used.
        """,
    )
    scaled_range: Optional[List[float]] = Field(
        default=None,
        description="""
Can be used to set the range of the control values
after scaling (default = [0, 1]).

This option has no effect if auto_scale is not set.
        """,
    )
    sampler: Optional[SamplerConfig] = Field(
        default=None,
        description="""
A sampler specification section applies to a group of controls, or to an
individual control. Sampler specifications are not required, with the
following behavior, if no sampler sections are provided, a normal
distribution is used.

If at least one control group or variable has a sampler specification, only
the groups or variables with a sampler specification are perturbed.
Controls/variables that do not have a sampler section will not be perturbed
at all. If that is not desired, make sure to specify a sampler for each
control group and/or variable (or none at all to use a normal distribution
for each control).

Within the sampler section, the *shared* keyword can be used to direct the
sampler to use the same perturbations for each realization.
        """,
    )

    @field_validator("scaled_range")
    @classmethod
    def validate_is_range(cls, scaled_range):  # pylint: disable=E0213
        if len(scaled_range) != 2 or scaled_range[0] >= scaled_range[1]:
            raise ValueError("scaled_range must be a valid range [a, b], where a < b.")
        return scaled_range

    @model_validator(mode="after")
    def validate_variables(self) -> Self:
        error = []
        variables = self.variables
        if variables is None:
            return self
        if not variables:
            error.append("Empty variables data")
        has_index = [v.index is not None for v in variables]
        if any(has_index) and not all(has_index):
            error.append(
                "Index should be given either for all of the variables or for none"
                " of them"
            )

        namespace = [(v.name, v.index) for v in variables]
        if len(namespace) != len(set(namespace)):
            error.append("Variable name or name-index combination has to be unique")

        error.extend(
            chain(
                firgure_a_better_name(
                    variable.name,
                    self.min if variable.min is None else variable.min,
                    self.max if variable.max is None else variable.max,
                    self.initial_guess
                    if variable.initial_guess is None
                    else variable.initial_guess,
                )
                for variable in variables
            )
        )
        if error:
            raise ValueError(error)
        return self

    model_config = ConfigDict(
        extra="forbid",
    )
