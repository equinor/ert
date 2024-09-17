from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, NonNegativeInt, field_validator, model_validator

from everest.strings import DATE_FORMAT


class ModelConfig(BaseModel, extra="forbid"):  # type: ignore
    realizations: List[NonNegativeInt] = Field(
        default_factory=lambda: [],
        description="""List of realizations to use in optimization ensemble.

Typically, this is a list [0, 1, ..., n-1] of all realizations in the ensemble.""",
    )
    data_file: Optional[str] = Field(
        default=None,
        description="""Path to the eclipse data file used for optimization.
        The path can contain r{{geo_id}}.

NOTE: Without a data file no well or group specific summary data will be exported.""",
    )
    realizations_weights: Optional[List[float]] = Field(
        default=None,
        description="""List of weights, one per realization.

If specified, it must be a list of numeric values, one per realization.""",
    )
    report_steps: Optional[List[str]] = Field(
        default=None,
        description="List of dates allowed in the summary file.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_realizations_weights_same_cardinaltiy(cls, values):  # pylint: disable=E0213
        weights = values.get("realizations_weights")
        reals = values.get("realizations")

        if not weights:
            return values

        if len(weights) != len(reals):
            raise ValueError(
                "Specified realizations_weights must have one"
                " weight per specified realization in realizations"
            )

        return values

    @field_validator("report_steps")
    @classmethod
    def validate_report_steps_are_dates(cls, report_steps):  # pylint: disable=E0213
        invalid_steps = []
        for step in report_steps:
            try:
                if not isinstance(step, str):
                    invalid_steps.append(str(step))
                    continue

                datetime.strptime(step, DATE_FORMAT)
            except ValueError:
                invalid_steps.append(step)

        if len(invalid_steps) > 0:
            raise ValueError(
                f"malformed dates: {', '.join(invalid_steps)},"
                f"expected format: {DATE_FORMAT}"
            )

        return report_steps
