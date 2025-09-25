from textwrap import dedent

import numpy as np
from pydantic import BaseModel, Field, PositiveFloat, field_validator


class InputConstraintConfig(BaseModel, extra="forbid"):
    weights: dict[str, float] = Field(
        examples=[
            {
                "weights": [
                    {"point_3D.x.0": 1},
                    {"point_3D.y.1": 1},
                    {"point_3D.z.2": 1},
                ],
                "upper_bound": 0.2,
            }
        ],
        description=dedent(
            """
            The coefficients of the linear equation that defines the input constraint.

            Note: Although the term "weights" suggests that these coefficients
            must be normalized, or are internally normalized, this is not the
            case: they can take any value.
            """
        ),
    )
    target: float | None = Field(
        default=None,
        description=dedent(
            """
            Sets the right-hand-side value for a linear equality constraint.

            This will direct the optimizer to limit the control values during
            optimization to the set of values where the sum of the controls
            multiplied by their coefficients (`weights`) is equal to `target`.
            """
        ),
    )
    lower_bound: float = Field(
        default=-np.inf,
        description=dedent(
            """
            Sets the right-hand-side value for a linear equality constraint.

            This will direct the optimizer to limit the control values during
            optimization to the set of values where the sum of the controls
            multiplied by their coefficients (`weights`) is equal to, or greater
            than `lower_bound`.
            """
        ),
    )
    upper_bound: float = Field(
        default=np.inf,
        description=dedent(
            """
            Sets the right-hand-side value for a linear equality constraint.

            This will direct the optimizer to limit the control values during
            optimization to the set of values where the sum of the controls
            multiplied by their coefficients (`weights`) is equal to, or less
            than `upper_bound`.
            """
        ),
    )
    scale: PositiveFloat | None = Field(
        default=None,
        description=dedent(
            """
            Scaling of input constraints.

            `scale` is a normalization factor which can be used to scale the
            input constraint. The bounds or target, and the weights will be
            scaled with this number.

            This option will be disabled if `auto_scale` is set in the
            `optimization` section.
            """
        ),
    )

    @field_validator("weights")
    @classmethod
    def validate_weights_not_empty(cls, weights: dict[str, float]) -> dict[str, float]:
        if weights is None or weights == {}:
            raise ValueError("Input weight data required for input constraints")
        return weights
