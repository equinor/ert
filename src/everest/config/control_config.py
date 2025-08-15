import logging
from itertools import chain
from typing import (
    Annotated,
    Any,
    Literal,
    Self,
    TypeAlias,
)

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from ropt.enums import PerturbationType, VariableType

from ert.config import ExtParamConfig

from .control_variable_config import (
    ControlVariableConfig,
    ControlVariableGuessListConfig,
)
from .sampler_config import SamplerConfig
from .validation_utils import (
    control_variables_validation,
    no_dots_in_string,
    unique_items,
    valid_range,
)

ControlVariable: TypeAlias = (
    list[ControlVariableConfig] | list[ControlVariableGuessListConfig]
)

logger = logging.getLogger(__name__)


def _all_or_no_index(variables: ControlVariable) -> ControlVariable:
    if isinstance(variables[-1], ControlVariableGuessListConfig):
        return variables

    if len({getattr(variable, "index", None) is None for variable in variables}) != 1:
        raise ValueError(
            "Index should be given either for all of the variables or for none of them"
        )
    return variables


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
    variables: Annotated[
        ControlVariable,
        AfterValidator(_all_or_no_index),
        AfterValidator(unique_items),
    ] = Field(description="List of control variables", min_length=1)
    initial_guess: float | None = Field(
        default=None,
        description="""
Initial guess for the control group all control variables with initial_guess not
defined will be assigned this value. Individual initial_guess values in the control
variables will overwrite this value.
""",
    )
    control_type: Literal["real", "integer"] = Field(
        default="real",
        description="""
The type of the controls for the control group. Individual control types in the
control variables will override this value. Set to "integer" for discrete
optimization. This may be ignored if the algorithm that is used does not support
different control types.
""",
    )
    enabled: bool = Field(
        default=True,
        description="""
If `True`, all variables in this control group will be optimized. If set to `False`
the value of the variables will remain fixed.
""",
    )
    min: float | None = Field(
        default=None,
        description="""
Defines left-side value in the control group range [min, max].
This value will be overwritten by the control variable min value if given.

The initial guess for both the group and the individual variables needs to be contained
in the resulting [min, max] range
""",
    )
    max: float | None = Field(
        default=None,
        description="""
Defines right-side value in the control group range [min, max].
This value will be overwritten by the control variable max value if given.

The initial guess for both the group and the individual variables needs to be contained
in the resulting [min, max] range
""",
    )
    perturbation_type: Literal["absolute", "relative"] = Field(
        default="absolute",
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
    perturbation_magnitude: float | None = Field(
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
    scaled_range: Annotated[tuple[float, float], AfterValidator(valid_range)] = Field(
        default=(0.0, 1.0),
        description="""
Can be used to set the range of the control values
after scaling (default = [0, 1]).

This option has no effect on discrete controls.
        """,
    )
    sampler: SamplerConfig | None = Field(
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

    @property
    def ropt_perturbation_type(self) -> PerturbationType:
        return PerturbationType[self.perturbation_type.upper()]

    @property
    def ropt_control_type(self) -> VariableType:
        return VariableType[self.control_type.upper()]

    @model_validator(mode="before")
    @classmethod
    def check_for_autoscale_flag(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "auto_scale" in values:
            raise ValueError(
                "auto_scale is deprecated for everest controls, and is on by default."
            )
        return values

    @model_validator(mode="after")
    def validate_variables(self) -> Self:
        if self.variables is None:
            return self

        if error := list(
            chain.from_iterable(
                control_variables_validation(
                    variable.name,
                    self.min if variable.min is None else variable.min,
                    self.max if variable.max is None else variable.max,
                    (
                        self.initial_guess
                        if variable.initial_guess is None
                        else variable.initial_guess
                    ),
                )
                for variable in self.variables
            )
        ):
            raise ValueError(error)
        return self

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and other.name == self.name

    @property
    def uniqueness(self) -> str:
        return "name"

    @property
    def formatted_control_names(self) -> list[str]:
        formatted_names = []
        for variable in self.variables:
            if isinstance(variable, ControlVariableGuessListConfig):
                for index in range(1, len(variable.initial_guess) + 1):
                    formatted_names.append(f"{self.name}.{variable.name}.{index}")
            elif variable.index is not None:
                formatted_names.append(f"{self.name}.{variable.name}.{variable.index}")
            else:
                formatted_names.append(f"{self.name}.{variable.name}")

        return formatted_names

    @property
    def formatted_control_names_dotdash(self) -> list[str]:
        formatted_names = []
        for variable in self.variables:
            if isinstance(variable, ControlVariableGuessListConfig):
                for index in range(1, len(variable.initial_guess) + 1):
                    formatted_names.append(f"{self.name}.{variable.name}-{index}")
            elif variable.index is not None:
                formatted_names.append(f"{self.name}.{variable.name}-{variable.index}")
            else:
                formatted_names.append(f"{self.name}.{variable.name}")

        return formatted_names

    model_config = ConfigDict(
        extra="forbid",
    )

    def to_ert_parameter_config(self) -> ExtParamConfig:
        return ExtParamConfig(
            name=self.name,
            input_keys=self.formatted_control_names,
            output_file=self.name + ".json",
        )
