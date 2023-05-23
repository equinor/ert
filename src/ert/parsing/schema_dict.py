from typing import Any, Dict, List, Set

from .config_errors import ConfigValidationError
from .error_info import ErrorInfo


class SchemaItemDict(dict):
    def __init__(self) -> None:
        pass

    def check_required(
        self,
        config_dict: Dict[str, Any],
        filename: str,
    ) -> None:
        errors: List[ErrorInfo] = []

        deprecated_keyword_usages = [
            schema_item
            for schema_item in self.values()
            if schema_item.deprecated and schema_item.kw in config_dict
        ]
        for schema_item in deprecated_keyword_usages:
            errors.append(
                ErrorInfo(
                    filename=filename, message=schema_item.deprecate_msg
                ).set_context_keyword(config_dict[schema_item.kw])
            )

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

            if constraints.required_set and constraints.kw not in config_dict:
                errors.append(
                    ErrorInfo(
                        message=f"{constraints.kw} must be set.",
                        filename=filename,
                    )
                )

        if len(errors) > 0:
            raise ConfigValidationError(errors=errors)
