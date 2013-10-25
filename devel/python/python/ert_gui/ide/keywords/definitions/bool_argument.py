
import re
from ert_gui.ide.keywords.definitions import ArgumentDefinition


class BoolArgument(ArgumentDefinition):

    NOT_BOOL = "The argument must be a bool."

    pattern  = re.compile("^TRUE|FALSE$")

    def __init__(self, **kwargs):
        super(BoolArgument, self).__init__(**kwargs)


    def validate(self, token):
        validation_status = super(BoolArgument, self).validate(token)

        match = BoolArgument.pattern.match(token)

        if match is None:
            validation_status.setFailed()
            validation_status.addToMessage(BoolArgument.NOT_BOOL)
        else :
            value = True if "TRUE" == token else False

            validation_status.setValue(value)

        return validation_status







