from collections import defaultdict
from typing import List, Optional, Union

from ert.parsing.error_info import ErrorInfo


class ConfigWarning(UserWarning):
    pass


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
        super().__init__(
            ";".join(
                [self._get_old_style_error_message(error) for error in self.errors]
            )
        )

    @classmethod
    def from_info(cls, info: ErrorInfo) -> "ConfigValidationError":
        return cls([info])

    @classmethod
    def _get_old_style_error_message(cls, info: ErrorInfo) -> str:
        return (
            (
                f"Parsing config file `{info.filename}` "
                f"resulted in the errors: {info.message}"
            )
            if info.filename is not None
            else info.message
        )

    def get_cli_message(self) -> str:
        return "\n".join(self.get_error_messages())

    def get_error_messages(self) -> List[str]:
        by_filename = defaultdict(list)
        for error in self.errors:
            by_filename[error.filename].append(error)

        nice_messages = []
        for filename, info_list in by_filename.items():
            for info in info_list:
                nice_messages.append(
                    f"{filename}:"
                    f"{info.line}:"
                    f"{info.column}:{info.end_column}:"
                    f"{info.message}"
                )

        return nice_messages

    @classmethod
    def from_collected(
        cls, errors: List[Union[ErrorInfo, "ConfigValidationError"]]
    ) -> "ConfigValidationError":
        # Turn into list of only ConfigValidationErrors
        all_error_infos = []

        for e in errors:
            if isinstance(e, ConfigValidationError):
                all_error_infos += e.errors
            else:
                all_error_infos.append(e)

        return cls(all_error_infos)
