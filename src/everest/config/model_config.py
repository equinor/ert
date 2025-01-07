from pydantic import BaseModel, Field, NonNegativeInt, model_validator

from ert.config import ConfigWarning


class ModelConfig(BaseModel, extra="forbid"):  # type: ignore
    realizations: list[NonNegativeInt] = Field(
        description="""List of realizations to use in optimization ensemble.

Typically, this is a list [0, 1, ..., n-1] of all realizations in the ensemble.""",
        min_length=1,
    )
    data_file: str | None = Field(
        default=None,
        description="""Path to the eclipse data file used for optimization.
        The path can contain r{{geo_id}}.

NOTE: Without a data file no well or group specific summary data will be exported.""",
    )
    realizations_weights: list[float] | None = Field(
        default=None,
        description="""List of weights, one per realization.

If specified, it must be a list of numeric values, one per realization.""",
    )

    @model_validator(mode="before")
    @classmethod
    def remove_deprecated(cls, values):
        if values is None:
            return values

        if values.get("report_steps") is not None:
            ConfigWarning.warn(
                "report_steps no longer has any effect and can be removed."
            )
            values.pop("report_steps")
        return values

    @model_validator(mode="after")
    def validate_realizations_weights_same_cardinaltiy(self):  # pylint: disable=E0213
        weights = self.realizations_weights
        if not weights:
            return self

        if len(weights) != len(self.realizations):
            raise ValueError(
                "Specified realizations_weights must have one"
                " weight per specified realization in realizations"
            )
        return self
