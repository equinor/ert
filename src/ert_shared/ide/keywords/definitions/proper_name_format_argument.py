import re
from ert_shared.ide.keywords.definitions import ArgumentDefinition


class ProperNameFormatArgument(ArgumentDefinition):

    NOT_A_VALID_NAME_FORMAT = (
        "The argument must be a valid string containing a %d and only characters of these types: "
        "Letters: A-Z and a-z, "
        "numbers: 0-9, "
        "underscore: _, "
        "dash: -, "
        "period: . and "
        "brackets: > < "
    )

    PATTERN = re.compile(r"^[A-Za-z0-9_\-.<>]*(%d)[A-Za-z0-9_\-.<>]*$")

    def __init__(self, **kwargs):
        super(ProperNameFormatArgument, self).__init__(**kwargs)

    def validate(self, token):
        validation_status = super(ProperNameFormatArgument, self).validate(token)

        if not validation_status:
            return validation_status
        else:
            match = ProperNameFormatArgument.PATTERN.match(token)

            if match is None:
                validation_status.setFailed()
                validation_status.addToMessage(
                    ProperNameFormatArgument.NOT_A_VALID_NAME_FORMAT
                )
            else:

                if not validation_status.failed():
                    validation_status.setValue(token)

            return validation_status
