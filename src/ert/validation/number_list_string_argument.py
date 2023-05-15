import re

from .argument_definition import ArgumentDefinition
from .validation_status import ValidationStatus


class NumberListStringArgument(ArgumentDefinition):
    NOT_A_VALID_NUMBER_LIST_STRING = (
        "The input should be of the type: "
        "<b><pre>\n\t23,5.5,11,1.01,3\n</pre></b>"
        "i.e. numeric values separated by commas."
    )
    VALUE_NOT_A_NUMBER = "The value: '%s' is not a number."

    PATTERN = re.compile(r"^[0-9\.\-+, \t]+$")

    def __init__(self, **kwargs: bool) -> None:
        super().__init__(**kwargs)

    def validate(self, token: str) -> ValidationStatus:
        validation_status = super().validate(token)

        if not validation_status:
            return validation_status
        else:
            match = NumberListStringArgument.PATTERN.match(token)

            if match is None:
                validation_status.setFailed()
                validation_status.addToMessage(
                    NumberListStringArgument.NOT_A_VALID_NUMBER_LIST_STRING
                )
            else:
                groups = token.split(",")

                for group in groups:
                    group = group.strip()

                    if len(group) > 0:
                        try:
                            float(group.strip())
                        except ValueError:
                            validation_status.setFailed()
                            validation_status.addToMessage(
                                NumberListStringArgument.VALUE_NOT_A_NUMBER % group
                            )

                validation_status.setValue(token)

            return validation_status
