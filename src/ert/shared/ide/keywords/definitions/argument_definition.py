from ert.shared.ide.keywords.data import ValidationStatus


class ArgumentDefinition:
    MISSING_ARGUMENT = "Missing argument!"

    def __init__(self, optional=False, built_in=False, rest_of_line=False) -> None:
        super().__init__()
        self.__optional = optional
        self.__built_in = built_in
        self.__rest_of_line = rest_of_line

    def isOptional(self):
        return self.__optional

    def isBuiltIn(self):
        return self.__built_in

    def consumeRestOfLine(self):
        return self.__rest_of_line

    def validate(self, token) -> ValidationStatus:
        vs = ValidationStatus()

        if not self.isOptional() and token.strip() == "":
            vs.setFailed()
            vs.addToMessage(ArgumentDefinition.MISSING_ARGUMENT)

        return vs
