from typing import Optional

from pydantic import BaseModel, Field, PositiveFloat, field_validator


class ObjectiveFunctionConfig(BaseModel, extra="forbid"):  # type: ignore
    name: str = Field()
    alias: Optional[str] = Field(
        default=None,
        description="""
alias can be set to the name of another objective function, directing everest
to copy the value of that objective into the current objective. This is useful
when used together with the **type** option, for instance to construct an objective
function that consist of the sum of the mean and standard-deviation over the
realizations of the same objective. In such a case, add a second objective with
**type** equal to "stddev" and set **alias** to the name of the first objective to make
sure that the standard deviation is calculated over the values of that objective.
""",
    )
    weight: Optional[PositiveFloat] = Field(
        default=None,
        description="""
weight determines the importance of an objective function relative to the other
objective functions.

Ultimately, the weighted sum of all the objectives is what Everest tries to optimize.
Note that, in case the weights do not sum up to 1, they are normalized before being
used in the optimization process.
""",
    )
    normalization: Optional[float] = Field(
        default=None,
        description="""
normalization is a multiplication factor defined per objective function.

The value of each objective function is multiplied by the related normalization value.
When optimizing with respect to multiple objective functions, it is important
that the normalization is set so that all the normalized objectives have the same order
of magnitude. Ultimately, the normalized objectives are used in computing
the weighted sum that Everest tries to optimize.
""",
    )
    auto_normalize: Optional[bool] = Field(
        default=None,
        description="""
auto_normalize can be set to true to automatically
determine the normalization factor from the objective value in batch 0.

If normalization is also set, the automatic value is multiplied by its value.
""",
    )
    type: Optional[str] = Field(
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

    @field_validator("normalization")
    @classmethod
    def validate_normalization_is_not_zero(cls, normalization):  # pylint: disable=E0213
        if normalization == 0.0:
            raise ValueError("Normalization value cannot be zero")
        return normalization
