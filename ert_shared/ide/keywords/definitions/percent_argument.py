import re
from ert_shared.ide.keywords.definitions import ArgumentDefinition
from ert_shared.ide.keywords.definitions import IntegerArgument


class PercentArgument(ArgumentDefinition):

    NOT_PERCENT = "The argument must be a number followed by % - no space allowed."
    NOT_IN_RANGE = "The argument is not in range: %s"

    pattern = re.compile(r"^-?[0-9]+(\.[0-9]+)?\%$")

    def __init__(self, from_value, to_value, **kwargs):
        super(PercentArgument, self).__init__(**kwargs)
        self.from_value = from_value * 0.01
        self.to_value = to_value * 0.01

    def validate(self, token):
        validation_status = super(PercentArgument, self).validate(token)

        match = PercentArgument.pattern.match(token)

        if match is None:
            validation_status.setFailed()
            validation_status.addToMessage(PercentArgument.NOT_PERCENT)
        else:
            value = float(token[:-1]) * 0.01

            if not self.from_value <= value <= self.to_value:
                validation_status.setFailed()
                range_string = "{:.0f}% <= {:.0f}% <= {:.0f}%".format(
                    self.from_value * 100, value * 100, self.to_value * 100
                )
                validation_status.addToMessage(
                    IntegerArgument.NOT_IN_RANGE % range_string
                )

            if not validation_status.failed():
                validation_status.setValue(value)

        return validation_status
