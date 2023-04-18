from collections import defaultdict
from textwrap import indent
from typing import List, Optional, Tuple, Union

from ert.parsing.lark_parser_error_info import ErrorInfo


class ConfigWarning(UserWarning):
    pass


class ConfigValidationError(ValueError):
    def __init__(
        self,
        errors: Union[str, List[Union[ErrorInfo, Tuple[Optional[str], str]]]],
        config_file: Optional[str] = None,
    ) -> None:
        self.errors: List[ErrorInfo] = []
        if isinstance(errors, list):
            for err in errors:
                if isinstance(err, str):
                    self.errors.append(ErrorInfo(message=err, filename=config_file))
                elif isinstance(err, ErrorInfo):
                    self.errors.append(err)
                else:
                    filename, error = err
                    self.errors.append(ErrorInfo(message=err, filename=filename))
        else:
            self.errors.append(ErrorInfo(message=errors, filename=config_file))
        super().__init__(
            ";".join([self._get_error_message(error) for error in self.errors])
        )

    @classmethod
    def from_info(cls, info: ErrorInfo):
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

    def get_error_messages(self):
        return [self._get_error_message(error) for error in self.errors]

    def get_cli_message(self):
        by_filename = defaultdict(list)
        for error in self.errors:
            by_filename[error.filename].append(error)

        return ";".join(
            [
                f"Parsing config file `{filename}` resulted in the errors: \n"
                + indent("\n".join([err.message for err in errors]), "  * ")
                for filename, errors in by_filename.items()
            ]
        )

    @classmethod
    def from_collected(cls, errors: List["ConfigValidationError"]):
        return cls([error_info for error in errors for error_info in error.errors])
