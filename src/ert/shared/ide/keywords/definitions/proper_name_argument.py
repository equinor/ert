import re

from ert.shared.ide.keywords.definitions import ArgumentDefinition


class ProperNameArgument(ArgumentDefinition):

    NOT_A_VALID_NAME = (
        "The argument must be a valid string "
        "containing only characters of these types: "
        "Letters: A-Z and a-z, "
        "numbers: 0-9, "
        "underscore: _, "
        "dash: -, "
        "period: . and "
        "brackets: > < "
    )

    PATTERN = re.compile(r"^[A-Za-z0-9_\-.<>]+$")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validate(self, token):
        validation_status = super().validate(token)

        if not validation_status:
            return validation_status
        else:
            match = ProperNameArgument.PATTERN.match(token)

            if match is None:
                validation_status.setFailed()
                validation_status.addToMessage(ProperNameArgument.NOT_A_VALID_NAME)
            else:

                if not validation_status.failed():
                    validation_status.setValue(token)

            return validation_status
