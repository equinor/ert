import abc
import warnings
from typing import Any, Dict, List, Set

from .config_errors import ConfigValidationError, ConfigWarning
from .config_schema_deprecations import DeprecationInfo
from .config_schema_item import SchemaItem
from .context_values import ContextList, ContextString
from .error_info import ErrorInfo, WarningInfo
from .types import ConfigDict


class SchemaItemDict(dict):
    def search_for_unset_required_keywords(
        self, config_dict: ConfigDict, filename: str
    ):
        errors = []
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

        if errors:
            raise ConfigValidationError.from_collected(errors)

    def add_deprecations(self, deprecated_keywords_list: List[DeprecationInfo]):
        deprecated_kws_not_in_schema = [
            info.keyword
            for info in deprecated_keywords_list
            if info.keyword not in self
        ]

        for kw in deprecated_kws_not_in_schema:
            # Add it to the schema only so that it is
            # catched by the parser
            self[kw] = SchemaItem.deprecated_dummy_keyword(kw)

    def search_for_deprecated_keyword_usages(
        self,
        config_dict: ConfigDict,
        filename: str,
        deprecated_keywords_list: List[DeprecationInfo],
    ):
        detected_deprecations = []
        maybe_deprecated_kws_dict = {x.keyword: x for x in deprecated_keywords_list}

        def push_deprecation(info: DeprecationInfo, line: List[ContextString]):
            if info.check is None or (callable(info.check) and info.check(line)):
                detected_deprecations.append((info, line))

        for kw, v in config_dict.items():
            deprecation_info = maybe_deprecated_kws_dict.get(kw)
            if deprecation_info and kw in self:
                if v is None:
                    # Edge case:
                    continue

                if isinstance(v, ContextString):
                    push_deprecation(
                        deprecation_info,
                        ContextList.with_values(token=v.keyword_token, values=[v]),
                    )
                elif isinstance(v, list) and (
                    len(v) == 0 or isinstance(v[0], ContextString)
                ):
                    push_deprecation(deprecation_info, v)
                elif isinstance(v[0], list):
                    for arglist in v:
                        push_deprecation(deprecation_info, arglist)

        if detected_deprecations:
            for deprecation, line in detected_deprecations:
                warnings.warn(
                    ConfigWarning(
                        WarningInfo(
                            is_deprecation=True,
                            filename=filename,
                            message=deprecation.resolve_message(line),
                        ).set_context_keyword(line)
                    )
                )

    @abc.abstractmethod
    def check_required(
        self,
        config_dict: Dict[str, Any],
        filename: str,
    ) -> None:
        pass
