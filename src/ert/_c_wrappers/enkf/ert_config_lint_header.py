from dataclasses import dataclass
from enum import Enum
from typing import Optional

import lark.tree

from ert._c_wrappers.config import ConfigValidationError


class LintType(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class LintLocation:
    line: int
    end_line: int
    column: int
    end_column: int

    @classmethod
    def from_meta(cls, meta: lark.tree.Meta):
        return cls(meta.line, meta.end_line, meta.column, meta.end_column)

    def __eq__(self, other):
        return (
            self.line == other.line
            and self.end_line == other.end_line
            and self.column == other.column
            and self.end_column == other.end_column
        )


@dataclass
class LintInfo:
    message: Optional[str]
    lint_type: LintType
    location: LintLocation


class ErtConfigLinter:
    def __init__(self):
        self.lints: [LintInfo] = []

    @classmethod
    def from_config_validation_error(cls, e: ConfigValidationError):
        pass

    def add_lint(
        self,
        lint_type: LintType,
        message: str,
        meta: Optional[lark.tree.Meta] = None,
        line: Optional[int] = None,
        end_line: Optional[int] = None,
        column: Optional[int] = None,
        end_column: Optional[int] = None,
    ):
        location: LintLocation = None
        if meta:
            location = LintLocation.from_meta(meta)
        else:
            location = LintLocation(
                line=line,
                end_line=end_line,
                column=column,
                end_column=end_column,
            )

        self.lints.append(
            LintInfo(
                message=message,
                lint_type=lint_type,
                location=location,
            )
        )

    def is_empty(self) -> bool:
        return len(self.lints) == 0

    def number_of_lints(self) -> int:
        return len(self.lints)

    def get(self, lint_index: int) -> Optional[LintInfo]:
        return self.lints[lint_index]
