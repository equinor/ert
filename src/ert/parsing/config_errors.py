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
            ";".join([self.get_value_error_message(error) for error in self.errors])
        )

    @classmethod
    def from_info(cls, info: ErrorInfo) -> "ConfigValidationError":
        return cls([info])

    @classmethod
    def get_value_error_message(cls, info: ErrorInfo) -> str:
        """
        :returns: The error message as used by cls as a ValueError.
        Can be overridden.
        """
        return (
            (
                f"Parsing config file `{info.filename}` "
                f"resulted in the following errors: {info.message}"
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

        messages = []
        for filename, info_list in by_filename.items():
            for info in info_list:
                messages.append(
                    ":".join(
                        [
                            str(k)
                            for k in [
                                filename,
                                info.line,
                                info.column,
                                info.end_column,
                                info.message,
                            ]
                            if k is not None
                        ]
                    )
                )

        return messages

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
