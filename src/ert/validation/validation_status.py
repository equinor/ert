class ValidationStatus:
    def __init__(self) -> None:
        super().__init__()
        self.__fail = False
        self.__message = ""
        self.__value: str | None = None

    def setFailed(self) -> None:
        self.__fail = True

    def failed(self) -> bool:
        return self.__fail

    def addToMessage(self, message: str) -> None:
        self.__message += message + "\n"

    def message(self) -> str:
        return self.__message.strip()

    def setValue(self, value: str) -> None:
        self.__value = value

    def value(self) -> str | None:
        return self.__value

    def __bool__(self) -> bool:
        return not self.__fail

    def __str__(self) -> str:
        return self.__message
