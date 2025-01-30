from __future__ import annotations

from .active_range import ActiveRange
from .argument_definition import ArgumentDefinition
from .validation_status import ValidationStatus


class RangeStringArgument(ArgumentDefinition):
    NOT_A_VALID_RANGE_STRING = (
        "The input should be of the type: "
        "<b><pre>\n\t1,3-5,9,17\n</pre></b>"
        "i.e. integer values separated by commas, and dashes to represent ranges."
    )
    VALUE_NOT_IN_RANGE = "A value must be in the range from 0 to %d."

    def __init__(self, max_value: int | None = None, **kwargs: bool) -> None:
        super().__init__(**kwargs)
        self.__max_value = max_value

    def validate(self, token: str) -> ValidationStatus:
        validation_status = super().validate(token)

        if not validation_status:
            return validation_status

        try:
            ActiveRange.validate_rangestring(token)
        except ValueError:
            validation_status.setFailed()
            validation_status.addToMessage(RangeStringArgument.NOT_A_VALID_RANGE_STRING)

        if self.__max_value is not None:
            try:
                ActiveRange.validate_rangestring_vs_length(token, self.__max_value)
            except ValueError:
                validation_status.setFailed()
                validation_status.addToMessage(
                    RangeStringArgument.VALUE_NOT_IN_RANGE % (self.__max_value - 1)
                )

        validation_status.setValue(token)

        return validation_status


class RangeSubsetStringArgument(RangeStringArgument):
    def __init__(self, source_active_range: ActiveRange) -> None:
        super().__init__()
        self._source_range = source_active_range

    def validate(self, token: str) -> ValidationStatus:
        validation_status = super().validate(token)

        if not validation_status:
            return validation_status
        try:
            other_range = ActiveRange(rangestring=token, length=len(self._source_range))
            self._source_range.validate_range_is_subset(other_range)
        except ValueError as e:
            validation_status.setFailed()
            validation_status.addToMessage(str(e))

        validation_status.setValue(token)
        return validation_status
