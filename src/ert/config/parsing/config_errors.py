from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Sequence, Type, Union

from typing_extensions import Self

from .error_info import ErrorInfo, WarningInfo
from .types import MaybeWithContext


class ConfigWarning(UserWarning):
    info: WarningInfo

    @staticmethod
    def ert_context_warn(message: str, context: MaybeWithContext = "") -> None:
        ConfigWarning.ert_formatted_warn(ConfigWarning.with_context(message, context))

    @staticmethod
    def ert_formatted_warn(config_warning: ConfigWarning) -> None:
        temp = warnings.formatwarning

        def ert_formatted_warning(
            message: Union[Warning, str],
            category: Type[Warning],
            filename: str,
            lineno: int,
            line: Union[str, None] = None,
        ) -> str:
            return str(message) + "\n"

        warnings.formatwarning = ert_formatted_warning
        warnings.warn(config_warning, stacklevel=1)
        warnings.formatwarning = temp

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
    """Contains one or more configuration errors to be shown to the user."""

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
    def with_context(cls, msg: str, context: MaybeWithContext) -> Self:
        """
        Create a single `ConfigValidationError` with some potential context
        (location in a file, line number etc.) with the given message.
        """
        if isinstance(context, List):
            return cls.from_info(ErrorInfo(msg).set_context_list(context))
        else:
            return cls.from_info(ErrorInfo(msg).set_context(context))

    @classmethod
    def from_info(cls, info: ErrorInfo) -> Self:
        return cls([info])

    def cli_message(self) -> str:
        """the configuration error messages as suitable for printing to cli"""
        return "\n".join(self.messages())

    def messages(self, formatter: Callable[[ErrorInfo], str] = str) -> List[str]:
        """List of the configuration errors messages with context"""
        return [formatter(info) for info in sorted(self.errors)]

    @classmethod
    def from_collected(
        cls, errors: Sequence[Union[ErrorInfo, ConfigValidationError]]
    ) -> Self:
        """Combine a list of ConfigValidationErrors (or ErrorInfo) into one.

        This is done so that the user can get get shown more than one error at
        the time, if there are more. This is opposed to stopping at the first
        error found, which is not ergonomic for resolving issues with the
        configuration files.
        """
        # Turn into list of only ConfigValidationErrors
        all_error_infos = []

        for e in errors:
            if isinstance(e, ConfigValidationError):
                all_error_infos += e.errors
            else:
                all_error_infos.append(e)

        return cls(all_error_infos)
