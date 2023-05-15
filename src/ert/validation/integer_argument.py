import re
from typing import Optional

from .argument_definition import ArgumentDefinition
from .validation_status import ValidationStatus


class IntegerArgument(ArgumentDefinition):
    NOT_INTEGER = "The argument must be an integer."
    NOT_IN_RANGE = "The argument is not in range: %s"

    pattern = re.compile("^-?[0-9]+$")

    def __init__(
        self,
        from_value: Optional[int] = None,
        to_value: Optional[int] = None,
        **kwargs: bool,
    ) -> None:
        super().__init__(**kwargs)
        self.from_value = from_value
        self.to_value = to_value

    def validate(self, token: str) -> ValidationStatus:
        validation_status = super().validate(token)

        match = IntegerArgument.pattern.match(token)

        if match is None:
            validation_status.setFailed()
            validation_status.addToMessage(IntegerArgument.NOT_INTEGER)
        else:
            value = int(token)

            if (
                self.from_value is not None
                and self.to_value is not None
                and not self.from_value <= value <= self.to_value
            ):
                validation_status.setFailed()
                range_string = f"{self.from_value} <= {value} <= {self.to_value}"
                validation_status.addToMessage(
                    IntegerArgument.NOT_IN_RANGE % range_string
                )

            elif self.from_value is not None and self.from_value > value:
                validation_status.setFailed()
                range_string = f"{self.from_value} <= {value}"
                validation_status.addToMessage(
                    IntegerArgument.NOT_IN_RANGE % range_string
                )

            elif self.to_value is not None and self.to_value < value:
                validation_status.setFailed()
                range_string = f"{value} <= {self.to_value}"
                validation_status.addToMessage(
                    IntegerArgument.NOT_IN_RANGE % range_string
                )

            if not validation_status.failed():
                validation_status.setValue(token)

        return validation_status
