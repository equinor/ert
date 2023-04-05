from typing import Optional


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