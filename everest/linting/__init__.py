"""The everest.linting module offers linting functionality.

Exposes the validator decorator.

"""

from .lintmessage import LintMessage
from .validation import validator

__all__ = [
    "validator",
    "LintMessage",
]
