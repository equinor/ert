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
            ";".join([self._get_error_message(error) for error in self.errors])
        )

    @classmethod
    def from_info(cls, info: ErrorInfo) -> "ConfigValidationError":
        return cls([info])

    @classmethod
    def _get_error_message(cls, info: ErrorInfo) -> str:
        return (
            (
                f"Parsing config file `{info.filename}` "
                f"resulted in the errors: {info.message}"
            )
            if info.filename is not None
            else info.message
        )

    def get_error_messages(self) -> List[str]:
        return [self._get_error_message(error) for error in self.errors]

    def get_cli_message(self) -> str:
        by_filename = defaultdict(list)
        for error in self.errors:
            by_filename[error.filename].append(error)

        def indent_only_first_line(message: str, prefix="  * "):
            lines = message.splitlines()
            blank_prefix = " " * len(prefix)

            return "\n".join(
                [prefix + lines[0], *[(blank_prefix + line) for line in lines[1:]]]
            )

        result = ";".join(
            [
                f"Parsing config file `{filename}` resulted in the errors: \n"
                + "\n".join(
                    [
                        indent_only_first_line(err.message_with_location, "  * ")
                        for err in info_list
                    ]
                )
                for filename, info_list in by_filename.items()
            ]
        )

        return result

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
