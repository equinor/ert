import abc
from collections import UserDict
from typing import no_type_check

from .config_dict import ConfigDict
from .config_errors import ConfigValidationError, ConfigWarning
from .config_schema_item import SchemaItem
from .context_values import ContextList, ContextString
from .deprecation_info import DeprecationInfo
from .error_info import ErrorInfo


class SchemaItemDict(UserDict[str, SchemaItem]):
    def search_for_unset_required_keywords(
        self, config_dict: ConfigDict, filename: str
    ) -> None:
        errors = []
        # schema.values()
        # can return duplicate values due to aliases
        # so we need to run this keyed by the keyword itself
        # Ex: there is an alias for NUM_REALIZATIONS
        # NUM_REALISATIONS
        # both with the same value
        # which causes .values() to return the NUM_REALIZATIONS keyword twice
        # which again leads to duplicate collection of errors related to this
        visited: set[str] = set()

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

    def add_deprecations(self, deprecated_keywords_list: list[DeprecationInfo]) -> None:
        for info in deprecated_keywords_list:
            # Add it to the schema only so that it is
            # catched by the parser
            if info.keyword not in self:
                self[info.keyword] = SchemaItem.deprecated_dummy_keyword(info)
            else:
                self[info.keyword].deprecation_info.append(info)

    @no_type_check
    def search_for_deprecated_keyword_usages(
        self,
        config_dict: ConfigDict,
        filename: str,
    ) -> None:
        detected_deprecations = []

        def push_deprecation(infos: list[DeprecationInfo], line: list[ContextString]):
            for info in infos:
                if info.check is None or (callable(info.check) and info.check(line)):
                    detected_deprecations.append((info, line))

        for kw, v in config_dict.items():
            schema_info = self.get(kw)
            if schema_info is not None and len(schema_info.deprecation_info) > 0:
                match v:
                    case None:
                        # Edge case: Happens if
                        # a keyword is specified in the schema and takes N args
                        # and is also specified as deprecated,
                        # and is specified in the config with 0 arguments
                        # which parses to None for the args
                        continue

                    case ContextString():
                        push_deprecation(
                            schema_info.deprecation_info,
                            ContextList.with_values(token=v.keyword_token, values=[v]),
                        )
                    case [ContextString(), *_]:
                        push_deprecation(schema_info.deprecation_info, v)
                    case [list(), *_]:
                        for arglist in v:
                            push_deprecation(schema_info.deprecation_info, arglist)
        if detected_deprecations:
            for deprecation, line in detected_deprecations:
                ConfigWarning.deprecation_warn(deprecation.resolve_message(line), line)

    @abc.abstractmethod
    def check_required(
        self,
        config_dict: ConfigDict,
        filename: str,
    ) -> None:
        """
        check_required checks that all required keywords are in config_dict and
        raises an ConfigValidationError if it is not.
        """
