import os

from .argument_definition import ArgumentDefinition
from .validation_status import ValidationStatus


class RunPathArgument(ArgumentDefinition):
    INVALID_PATH = "The specified runpath does not exist."
    MISSING_PERMISSION = "You are missing permissions for the specified runpath."

    def __init__(self, **kwargs: bool) -> None:
        super().__init__(**kwargs)

    def validate(self, token: str) -> ValidationStatus:
        parsed_runpath_without_suffix = "/".join(token.split("/")[:-2])
        validation_status = super().validate(token)

        if not os.path.isdir(parsed_runpath_without_suffix):
            validation_status.setFailed()
            validation_status.addToMessage(RunPathArgument.INVALID_PATH)
        elif not os.access(parsed_runpath_without_suffix, os.R_OK | os.X_OK):
            validation_status.setFailed()
            validation_status.addToMessage(RunPathArgument.MISSING_PERMISSION)
        else:
            validation_status.setValue(token)

        return validation_status
