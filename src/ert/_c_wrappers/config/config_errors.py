from dataclasses import dataclass
from typing import List, Optional, Union


class ExtJobInvalidArgsException(BaseException):
    pass


class ConfigWarning(UserWarning):
    pass


@dataclass()
class Location:
    filename: str
    start_pos: Optional[int] = None
    line: Optional[int] = None
    column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    end_pos: Optional[int] = None


class ConfigValidationError(ValueError):
    def __init__(
        self,
        errors: str,
        config_file: Optional[str] = None,
        location: Optional[Union[str, Location]] = None,
    ) -> None:
        if config_file:
            self.location = Location(config_file)
        elif location is not None:
            if isinstance(location, Location):
                self.location = location
            else:
                self.location = Location(location)
        else:
            self.location = Location(filename="")

        self.errors = errors

        super().__init__(
            (
                f"Parsing config file `{self.config_file}` "
                f"resulted in the errors: {self.errors}"
            )
            if self.config_file
            else f"{self.errors}"
        )

    def replace(self, old_text: str, new_text: str):
        return ConfigValidationError(
            errors=self.errors.replace(old_text, new_text),
            config_file=self.config_file,
            location=self.location,
        )

    @property
    def config_file(self):
        return self.location.filename

    @config_file.setter
    def config_file(self, config_file):
        self.location.filename = config_file

    def get_error_messages(self):
        return [self.errors]


class CombinedConfigError(ConfigValidationError):
    def __init__(
        self,
        errors: Optional[
            List[Union[ConfigValidationError, "CombinedConfigError"]]
        ] = None,
    ):
        self.errors_list = []

        for err in errors or []:
            self.add_error(err)

    @property
    def errors(self):
        return self.errors_list

    def __str__(self):
        return ", ".join(str(x) for x in self.errors_list)

    def is_empty(self):
        return len(self.errors_list) == 0

    def add_error(self, error: Union[ConfigValidationError, "CombinedConfigError"]):
        if isinstance(error, CombinedConfigError):
            self.errors_list.append(*error.errors_list)
        else:
            self.errors_list.append(error)

    def get_error_messages(self):
        all_messages = []
        for e in self.errors_list:
            all_messages.append(*e.get_error_messages())

        return all_messages

    def find_matching_error(self, match: str) -> Optional[ConfigValidationError]:
        return next(x for x in self.errors_list if match in str(x))

    @property
    def config_file(self):
        return self.errors_list[0].location.filename


class ObservationConfigError(ConfigValidationError):
    def __init__(self, errors: str, config_file: Optional[str] = None) -> None:
        super().__init__(
            errors=(
                f"Parsing observations config file `{config_file}` "
                f"resulted in the errors: {errors}"
            )
            if config_file
            else f"{errors}",
            config_file=config_file,
        )
