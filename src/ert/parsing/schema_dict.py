from typing import List, Set

from .config_errors import ConfigValidationError
from .error_info import ErrorInfo


class SchemaItemDict(dict):
    def __init__(self) -> None:
        pass

    def check_required(
        self,
        declared_kws: Set[str],
        filename: str,
    ) -> None:
        errors: List[ErrorInfo] = []

        # schema.values()
        # can return duplicate values due to aliases
        # so we need to run this keyed by the keyword itself
        # Ex: there is an alias for NUM_REALIZATIONS
        # NUM_REALISATIONS
        # both with the same value
        # which causes .values() to return the NUM_REALIZATIONS keyword twice
        # which again leads to duplicate collection of errors related to this
        visited: Set[str] = set()

        for constraints in self.values():
            if constraints.kw in visited:
                continue

            visited.add(constraints.kw)

            if constraints.required_set and constraints.kw not in declared_kws:
                errors.append(
                    ErrorInfo(
                        message=f"{constraints.kw} must be set.",
                        filename=filename,
                    )
                )

        if len(errors) > 0:
            raise ConfigValidationError(errors=errors)
