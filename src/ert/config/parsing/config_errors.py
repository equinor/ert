from typing import Callable, List, Optional, Sequence, Union

from .error_info import ErrorInfo, WarningInfo


class ConfigWarning(UserWarning):
    info: WarningInfo

    def __init__(self, info: Union[str, WarningInfo]):
        if isinstance(info, str):
            super().__init__(info)
            self.info = WarningInfo(message=info)
        else:
            super().__init__(info.message)
            self.info = info

    @classmethod
    def with_context(cls, msg: str, context: MaybeWithContext) -> Self:
        return cls(WarningInfo(msg).set_context(context))

    def __str__(self) -> str:
        return str(self.info)


class ConfigValidationError(ValueError):
    def __init__(
        self,
        errors: Union[str, List[ErrorInfo]],
        config_file: Optional[str] = None,
    ) -> None:
        self.errors: List[ErrorInfo] = []
        if isinstance(errors, list):
            for err in errors:
                if isinstance(err, ErrorInfo):
                    self.errors.append(err)
        else:
            self.errors.append(ErrorInfo(message=errors, filename=config_file))
        super().__init__(";".join([str(error) for error in self.errors]))

    @classmethod
    def from_info(cls, info: ErrorInfo) -> "ConfigValidationError":
        return cls([info])

    def get_cli_message(self) -> str:
        return "\n".join(self.get_error_messages())

    def get_error_messages(
        self, formatter: Callable[[ErrorInfo], str] = str
    ) -> List[str]:
        return [formatter(info) for info in sorted(self.errors)]

    @classmethod
    def from_collected(
        cls, errors: Sequence[Union[ErrorInfo, "ConfigValidationError"]]
    ) -> "ConfigValidationError":
        # Turn into list of only ConfigValidationErrors
        all_error_infos = []

        for e in errors:
            if isinstance(e, ConfigValidationError):
                all_error_infos += e.errors
            else:
                all_error_infos.append(e)

        return cls(all_error_infos)
