from textwrap import dedent
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    PositiveFloat,
    model_validator,
)

from everest.config.validation_utils import not_in_reserved_word_list


class ObjectiveFunctionConfig(BaseModel, extra="forbid"):
    name: Annotated[str, AfterValidator(not_in_reserved_word_list)] = Field(
        description=dedent(
            """
            The name of the objective function.
            """
        )
    )
    weight: float | None = Field(
        default=None,
        description=dedent(
            """
            The weight determines the importance of an objective function relative
            to the other objective functions.

            EVEREST optimizes the weighted sum of all objectives. The weights of
            all objectives are normalized to a total value of one by dividing
            them by their sum. Weights may be zero or negative, but should add
            up to a positive value.
            """
        ),
    )
    scale: PositiveFloat | None = Field(
        default=None,
        description=dedent(
            """
            Optional scaling of the objective function value.

            Each objective function will be divided by this scale value. This
            can be used to change the overall range of the objective values. For
            instance, this may be needed to properly balance the relative scales
            of objectives and output constraints.

            Note: This option should be used with care in case of
            multi-objective optimization, since scaling each objective
            differently will change their relative weights.

            This option will be disabled if `auto_scale` is set in the
            `optimization` section.
            """
        ),
    )
    type: Literal["mean", "stddev"] = Field(
        default="mean",
        description=dedent(
            """
            How to calculate the objective from realizations.

            The objective function value is aggregated from their individual
            realization values, usually by averaging them. The calculation used
            for this operation can be modified by setting this field. Currently,
            the two values are supported:

            - `"mean"` (default): calculates the mean over the realizations.
            - `"stddev"`: calculates the negative of the standard deviation over
              the realizations. The negative is used since in general the aim is
              to minimize the standard deviation (as opposed to the mean, which
              is preferred to be maximized).
            """
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def deprecate_normalization(cls, values: dict[str, Any]) -> dict[str, Any]:
        errors = []
        for key, replace in (
            ("normalization", "scale"),
            ("auto_normalize", "auto_scale in the optimization section"),
            ("auto_scale", "auto_scale in the optimization section"),
        ):
            if key in values:
                errors.append(
                    f"{key} is deprecated and has been replaced with {replace}"
                )
        if errors:
            raise ValueError(errors)
        return values
