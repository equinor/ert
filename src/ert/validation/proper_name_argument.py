from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .argument_definition import ArgumentDefinition
from .validation_status import ValidationStatus

if TYPE_CHECKING:
    from ..storage import Storage


class ProperNameArgument(ArgumentDefinition):
    NOT_A_VALID_NAME = (
        "The argument must be a valid string "
        "containing only characters of these types: "
        "Letters: A-Z and a-z, "
        "numbers: 0-9, "
        "underscore: _, "
        "dash: -, "
        "period: . and "
        "brackets: > < "
    )

    PATTERN = re.compile(r"^[A-Za-z0-9_\-.<>]+$")

    def __init__(self, **kwargs: bool) -> None:
        super().__init__(**kwargs)

    def validate(self, token: str) -> ValidationStatus:
        validation_status = super().validate(token)

        if not validation_status:
            return validation_status

        match = ProperNameArgument.PATTERN.match(token)

        if match is None:
            validation_status.setFailed()
            validation_status.addToMessage(ProperNameArgument.NOT_A_VALID_NAME)
        elif not validation_status.failed():
            validation_status.setValue(token)

        return validation_status


class ExperimentValidation(ProperNameArgument):
    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        super().__init__()

    def validate(self, token: str) -> ValidationStatus:
        validation_status = super().validate(token)

        if not validation_status:
            return validation_status

        existing = [exp.name for exp in self.storage.experiments]

        if token in existing:
            validation_status.setFailed()
            validation_status.addToMessage(
                f"Experiment name must be unique, not one of: {existing}"
            )
        elif not validation_status.failed():
            validation_status.setValue(token)

        return validation_status
