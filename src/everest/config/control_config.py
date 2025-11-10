import logging
from itertools import chain
from textwrap import dedent
from typing import (
    Annotated,
    Any,
    Literal,
    Self,
    TypeAlias,
)

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from ropt.enums import PerturbationType, VariableType

from ert.config import ExtParamConfig, SamplerConfig

from .control_variable_config import (
    ControlVariableConfig,
    ControlVariableGuessListConfig,
)
from .validation_utils import (
    control_variables_validation,
    no_dots_in_string,
    not_in_reserved_word_list,
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
    name: Annotated[
        str,
        AfterValidator(no_dots_in_string),
        AfterValidator(not_in_reserved_word_list),
    ] = Field(
        description=dedent(
            """
            The control group name.

            Controls name must be unique.
            """
        )
    )
    type: Literal["well_control", "generic_control"] = Field(
        description=dedent(
            r"""
            Control group type.

            Only two allowed control types are accepted:

            * `"well_control"`: Standard built-in Everest control type designed
              for field optimization
            * `"generic_control"`: Enables the user to define controls types to
              be employed for customized optimization jobs.
            """
        )
    )
    variables: Annotated[
        ControlVariable,
        AfterValidator(_all_or_no_index),
        AfterValidator(unique_items),
    ] = Field(
        description=dedent(
            """
            List of control variables.
            """
        ),
        min_length=1,
    )
    initial_guess: float | None = Field(
        default=None,
        description=dedent(
            """
            Initial guess for the control group.

            The initial guess value that is assigned to all control variables in
            the control group.

            This default value can be overridden at the variable level.
            """
        ),
    )
    control_type: Literal["real", "integer"] = Field(
        default="real",
        description=dedent(
            """
            Control data type for the control group.

            The data type value that is assigned to all control variables in
            the control group.

            Set to `"integer"` for discrete optimization, or `"real"` for
            continuous optimization (default). This may be ignored if the
            optimization algorithm that is used does not support this.

            This default value can be overridden at the variable level.
            """
        ),
    )
    enabled: bool = Field(
        default=True,
        description=dedent(
            """
            Enable/disable control groups.

            Whether all control variables in the control group are enabled or
            not. When not enabled, variables are kept constant at the initial
            guess value during optimization.

            This default value can be overridden at the variable level.
            """
        ),
    )
    min: float | None = Field(
        default=None,
        description=dedent(
            """
            The minimum values of all control variables in the control group.

            The `initial_guess` field of this control group and of all of its
            variables must respect this minimum value.

            This default value can be overridden at the variable level.
            """
        ),
    )
    max: float | None = Field(
        default=None,
        description=dedent(
            """
            The maximum values of all control variables in the control group.

            The `initial_guess` field of this control group and of all of its
            variables must respect this maximum value.

            This default value can be overridden at the variable level.
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
            """
        ),
    )
    perturbation_magnitude: float = Field(
        description=dedent(
            """
            The magnitude of the perturbation of all control variables in the
            control group.

            This controls the magnitude of perturbations (e.g. the standard
            deviation in case of a normal distribution) of controls, used to
            approximate the gradient.

            The interpretation of this field depends on the value of the
            `perturbation_type` field:

            - `absolute`: The given value is used as-is.
            - `relative`: The perturbation magnitude is calculated by
              multiplying the given by value by the difference between the `max`
              and `min` fields.

            This default value can be overridden at the variable level.
            """
        ),
    )
    scaled_range: Annotated[tuple[float, float], AfterValidator(valid_range)] = Field(
        default=(0.0, 1.0),
        description=dedent(
            """
            The target range of the control variable scaling for the control group.

            Internally control variables are scaled from their minimal and
            maximum values (`min`/`max` settings) to the range given by
            `target_range` (default = [0, 1]).

            This default value can be overridden at the variable level.

            This option has no effect on discrete controls.
            """
        ),
    )
    sampler: SamplerConfig | None = Field(
        default=None,
        description=dedent(
            """
            Sampler configuration for a control group.

            A sampler specification section applies to a group of controls, or
            to an individual control. Sampler specifications are not required,
            if no sampler sections are provided, a normal distribution is used.

            This can be overridden at the variable level.

            If at least one control group or variable has a sampler specification, only
            the groups or variables with a sampler specification are perturbed.
            Controls/variables that do not have a sampler section will not be perturbed
            at all. If that is not desired, make sure to specify a sampler for each
            control group and/or variable (or none at all to use a normal distribution
            for each control).

            Within the sampler section, the `shared` keyword can be used to
            direct the sampler to use the same perturbations for each
            realization.
            """
        ),
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
