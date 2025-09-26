from textwrap import dedent
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PositiveFloat, model_validator


class OutputConstraintConfig(BaseModel, extra="forbid"):
    name: str = Field(
        description=dedent(
            """
        The unique name of the output constraint.
        """
        )
    )
    target: float | None = Field(
        default=None,
        description=dedent(
            """
            Defines the equality constraint

            f(x) = b,

            where f is a function of the control vector x and b is the target.
            """
        ),
    )
    lower_bound: float = Field(
        default=-np.inf,
        description=dedent(
            """
            Defines the equality constraint

            f(x) >= b,

            where f is a function of the control vector x and b is the lower
            bound.
            """
        ),
    )
    upper_bound: float = Field(
        default=np.inf,
        description=dedent(
            """
            Defines the equality constraint

            f(x) <= b,

            where f is a function of the control vector x and b is the upper
            bound.
            """
        ),
    )
    scale: PositiveFloat | None = Field(
        default=None,
        description=dedent(
            """
            Scaling of constraints.

            `scale` is a normalization factor which can be used to scale the
            constraint to control its importance relative to the objective.

            Both the bounds or target and the function evaluation value will be
            scaled with this number. For example, if the upper bound is 0.5 and
            the scaling is 10, then the function evaluation value will be
            divided by 10 and bounded from above by 0.05.

            This option will be disabled if `auto_scale` is set in the
            `optimization` section.
            """
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def validate_target_or_bounds(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "target" in values and ("lower_bound" in values or "upper_bound" in values):
            raise ValueError("Can not combine target and bounds")
        elif not any(
            ("target" in values, "lower_bound" in values, "upper_bound" in values)
        ):
            raise ValueError("Must provide target or lower_bound/upper_bound")
        return values

    @model_validator(mode="before")
    @classmethod
    def deprecate_auto_scale(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "auto_scale" in values:
            raise ValueError(
                "auto_scale is deprecated and has been replaced with "
                "auto_scale in the optimization section"
            )
        return values
