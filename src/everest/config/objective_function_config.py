from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ObjectiveFunctionConfig(BaseModel, extra="forbid"):
    name: str = Field()
    weight: float | None = Field(
        default=None,
        description="""
weight determines the importance of an objective function relative to the other
objective functions.

Ultimately, the weighted sum of all the objectives is what Everest tries to optimize.
Note that, in case the weights do not sum up to 1, they are normalized before being
used in the optimization process. Weights may be zero or negative, but should add up to
a positive value.
""",
    )
    scale: float | None = Field(
        default=None,
        description="""
    scale is a division factor defined per objective function.

    The value of each objective function is divided by the related scaling value.
    When optimizing with respect to multiple objective functions, it is important
    that the scaling is set so that all the scaled objectives have the same order
    of magnitude. Ultimately, the scaled objectives are used in computing
    the weighted sum that Everest tries to optimize.
    """,
    )
    auto_scale: bool | None = Field(
        default=None,
        description="""
auto_normalize can be set to true to automatically
determine the scaling factor from the objective value in batch 0.

If scale is also set, the automatic value is divided by its value.
""",
    )
    type: str | None = Field(
        default=None,
        description="""
type can be set to the name of a method that should be applied to calculate a
total objective function from the objectives obtained for all realizations.
Currently, the only values supported are "mean" and "stddev", which calculate
the mean and the negative of the standard deviation over the realizations,
respectively. The negative of the standard deviation is used, since in general
the aim is to minimize the standard deviation as opposed to the mean, which is
preferred to be maximized.

""",
    )

    @field_validator("scale")
    @classmethod
    def validate_scale_is_not_zero(cls, scale: float | None) -> float | None:
        if scale == 0.0:
            raise ValueError("Scale value cannot be zero")
        return scale

    @model_validator(mode="before")
    @classmethod
    def deprecate_normalization(cls, values: dict[str, Any]) -> dict[str, Any]:
        errors = []
        for key, replace in (
            ("normalization", "scale"),
            ("auto_normalize", "auto_scale"),
        ):
            if key in values:
                errors.append(
                    f"{key} is deprecated and has been replaced with {replace}"
                )
        if errors:
            raise ValueError(errors)
        return values
