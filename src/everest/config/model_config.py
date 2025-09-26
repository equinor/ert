from collections.abc import Iterator
from textwrap import dedent
from typing import Any, Self

from pydantic import BaseModel, Field, NonNegativeInt, field_validator, model_validator

from ert.config import ConfigWarning


class ModelConfig(BaseModel, extra="forbid"):
    realizations: list[NonNegativeInt] = Field(
        description=dedent(
            """
            Realizations to use from the ensemble.

            This should be a list of realization ID's, indicating which
            realizations of the ensemble should be used.

            The list can be specified directly as a (YAML) list, or as a string
            consisting of comma separated values and ranges.

            For example `1, 2, 5-7, 10` is equivalent to `[1, 2, 5, 6, 7, 10]`.
            """
        ),
        min_length=1,
    )
    realizations_weights: list[float] = Field(
        default_factory=list,
        description=dedent(
            """
            A list of realization weights.

            If provided, the list must consist of one weight for each
            realization (as determined by the `realizations` field). If not
            provided, the weight of each realization is equal to one divided by
            the number or realizations.
            """
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def remove_deprecated(cls, values: dict[str, Any] | None) -> dict[str, Any] | None:
        if values is None:
            return None

        if values.get("report_steps") is not None:
            ConfigWarning.warn(
                "report_steps no longer has any effect and can be removed."
            )
            values.pop("report_steps")
        return values

    @field_validator("realizations", mode="before")
    @classmethod
    def convert_realizations(cls, realizations: Any) -> list[int]:
        def _parse_realizations(realizations: str) -> Iterator[int]:
            msg = f"Invalid realizations specification: {realizations}"
            for item in realizations.split(","):
                first, sep, second = (part.strip() for part in item.partition("-"))
                if not sep:
                    second = first
                if not first.isdigit() or not second.isdigit():
                    raise ValueError(msg)
                start, stop = int(first), int(second)
                if start > stop:
                    raise ValueError(msg)
                yield from range(start, stop + 1)

        if isinstance(realizations, str):
            return sorted(dict.fromkeys(_parse_realizations(realizations)))
        return realizations

    @model_validator(mode="after")
    def validate_realizations_weights_same_cardinaltiy(self) -> Self:
        weights = self.realizations_weights
        if not weights:
            self.realizations_weights = [1.0 / len(self.realizations)] * len(
                self.realizations
            )
            return self
        if len(weights) != len(self.realizations):
            raise ValueError(
                "Specified realizations_weights must have one"
                " weight per specified realization in realizations"
            )
        return self
