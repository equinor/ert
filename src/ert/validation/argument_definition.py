from .validation_status import ValidationStatus


class ArgumentDefinition:
    MISSING_ARGUMENT = "Missing argument!"

    def __init__(self, optional: bool = False) -> None:
        super().__init__()
        self.__optional = optional

    def isOptional(self) -> bool:
        return self.__optional

    def validate(self, token: str) -> ValidationStatus:
        vs = ValidationStatus()

        if not self.isOptional() and not token.strip():
            vs.setFailed()
            vs.addToMessage(ArgumentDefinition.MISSING_ARGUMENT)

        return vs
