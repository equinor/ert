from .validation_status import ValidationStatus


class StringDefinition:
    MISSING_TOKEN = "Missing required %s!"
    INVALID_TOKEN = "Contains invalid string %s!"

    def __init__(
        self,
        optional: bool = False,
        required: list[str] | None = None,
        invalid: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.__optional = optional
        self._required_tokens = required or []
        self._invalid_tokens = invalid or []

    def isOptional(self) -> bool:
        return self.__optional

    def validate(self, value: str) -> ValidationStatus:
        vs = ValidationStatus()
        required = [token for token in self._required_tokens if token not in value]
        invalid = [token for token in self._invalid_tokens if token in value]

        if not self.isOptional() and any(required):
            vs.setFailed()
            for token in required:
                vs.addToMessage(StringDefinition.MISSING_TOKEN % token)

        if not self.isOptional() and any(invalid):
            vs.setFailed()
            for token in invalid:
                vs.addToMessage(StringDefinition.INVALID_TOKEN % token)

        return vs
