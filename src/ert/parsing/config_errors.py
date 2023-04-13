from typing import List, Optional

from .lark_parser_error_info import ErrorInfo


class ConfigWarning(UserWarning):
    pass


class ConfigValidationError(ValueError):
    def __init__(self, errors: str, config_file: Optional[str] = None) -> None:
        self.config_file = config_file
        self.errors = errors
        super().__init__(
            (
                f"Parsing config file `{self.config_file}` "
                f"resulted in the errors: {self.errors}"
            )
            if self.config_file
            else f"{self.errors}"
        )

    @classmethod
    def raise_from_collected(cls, collected_errors: List[ErrorInfo]):
        if len(collected_errors) > 0:
            combined_str = ";".join([x.message for x in collected_errors])
            first_filename = next(
                x.filename for x in collected_errors if x.filename is not None
            )
            raise ConfigValidationError(
                errors=combined_str, config_file=first_filename or None
            )
