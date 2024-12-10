from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import cast
from warnings import catch_warnings

from .parsing import ConfigValidationError, ConfigWarning, ErrorInfo, WarningInfo


@dataclass
class ValidationMessages:
    warnings: list[WarningInfo] = field(default_factory=list)
    deprecations: list[WarningInfo] = field(default_factory=list)
    errors: list[ErrorInfo] = field(default_factory=list)


@contextmanager
def capture_validation() -> Iterator[ValidationMessages]:
    logger = logging.getLogger(__name__)
    validations = ValidationMessages()
    with catch_warnings(record=True) as all_warnings:
        try:
            yield validations
        except ConfigValidationError as err:
            validations.errors += err.errors

    for wm in all_warnings:
        if issubclass(wm.category, ConfigWarning):
            warning = cast(ConfigWarning, wm.message)
            if warning.info.is_deprecation:
                validations.deprecations.append(warning.info)
            else:
                validations.warnings.append(warning.info)
        else:
            logger.warning(str(wm.message))
