from pydantic import BaseModel, Field, model_validator


class OutputConstraintConfig(BaseModel, extra="forbid"):  # type: ignore
    name: str = Field(description="The unique name of the output constraint.")
    target: float | None = Field(
        default=None,
        description="""Defines the equality constraint

(f(x) - b) / c = 0,

where b is the target, f is a function of the control vector x, and c is the
scale (scale).

""",
    )
    auto_scale: bool | None = Field(
        default=None,
        description="""If set to true, Everest will automatically
determine the scaling factor from the constraint value in batch 0.

If scale is also set, the automatic value is multiplied by its value.""",
    )
    lower_bound: float | None = Field(
        default=None,
        description="""Defines the lower bound
(greater than or equal) constraint

(f(x) - b) / c >= 0,

where b is the lower bound, f is a function of the control vector x, and c is
the scale (scale).
""",
    )
    upper_bound: float | None = Field(
        default=None,
        description="""Defines the upper bound (less than or equal) constraint:

(f(x) - b) / c <= 0,

where b is the upper bound, f is a function of the control vector x, and c is
the scale (scale).""",
    )
    scale: float | None = Field(
        default=None,
        description="""Scaling of constraints (scale).

scale is a normalization factor which can be used to scale the constraint
to control its importance relative to the (singular) objective and the controls.

Both the upper_bound and the function evaluation value will be scaled with this number.
That means that if, e.g., the upper_bound is 0.5 and the scaling is 10, then the
function evaluation value will be divided by 10 and bounded from above by 0.05.

""",
    )

    @model_validator(mode="after")
    def validate_bounds_in_place(
        self,
    ):
        has_equality = self.target is not None
        has_inequality = self.upper_bound is not None or self.lower_bound is not None
        is_valid = has_equality ^ has_inequality

        if not is_valid:
            raise ValueError(
                "Output constraints must have only one of the following: { target },"
                " or { upper and/or lower bound }"
            )

        return self
