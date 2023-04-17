from collections import defaultdict
from textwrap import indent
from typing import List, Optional, Tuple, Union


class ConfigWarning(UserWarning):
    pass


class ConfigValidationError(ValueError):
    def __init__(
        self,
        errors: Union[str, List[Tuple[Optional[str], str]]],
        config_file: Optional[str] = None,
    ) -> None:
        self.errors: List[Tuple[Optional[str], str]] = []
        if isinstance(errors, list):
            for err in errors:
                if isinstance(err, str):
                    self.errors.append((config_file, err))
                else:
                    filename, error = err
                    self.errors.append((filename, error))
        else:
            self.errors.append((config_file, errors))
        super().__init__(
            ";".join(
                [
                    self._get_error_message(config_file, errors)
                    for config_file, errors in self.errors
                ]
            )
        )

    @classmethod
    def _get_error_message(cls, config_file: Optional[str], error: str) -> str:
        return (
            (f"Parsing config file `{config_file}` " f"resulted in the errors: {error}")
            if config_file is not None
            else error
        )

    def get_error_messages(self) -> List[str]:
        return [
            self._get_error_message(config_file, errors)
            for config_file, errors in self.errors
        ]

    def get_cli_message(self) -> str:
        by_filename = defaultdict(list)
        for filename, error in self.errors:
            by_filename[filename].append(error)
        return ";".join(
            [
                self._get_error_message(
                    config_file, "\n" + indent("\n".join(errors), "  * ")
                )
                for config_file, errors in by_filename.items()
            ]
        )

    @classmethod
    def from_collected(
        cls, errors: List["ConfigValidationError"]
    ) -> "ConfigValidationError":
        return cls(
            [
                (config_file, message)
                for error in errors
                for config_file, message in error.errors
            ]
        )
