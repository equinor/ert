import abc
from typing import TYPE_CHECKING, List, Set, no_type_check

from .config_dict import ConfigDict
from .config_errors import ConfigValidationError, ConfigWarning
from .config_schema_item import SchemaItem
from .context_values import ContextList, ContextString
from .deprecation_info import DeprecationInfo
from .error_info import ErrorInfo, WarningInfo

# Python 3.8 does not implement UserDict as a MutableMapping, meaning it's not
# possible to specify the key and value types.
#
# Instead, we only set the types during type checking
if TYPE_CHECKING:
    from collections import UserDict

    _UserDict = UserDict[str, SchemaItem]
else:
    from collections import UserDict as _UserDict


class SchemaItemDict(_UserDict):
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

    def add_deprecations(self, deprecated_keywords_list: List[DeprecationInfo]) -> None:
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

        def push_deprecation(infos: List[DeprecationInfo], line: List[ContextString]):
            for info in infos:
                if info.check is None or (callable(info.check) and info.check(line)):
                    detected_deprecations.append((info, line))

        for kw, v in config_dict.items():
            schema_info = self.get(kw)
            if schema_info is not None and len(schema_info.deprecation_info) > 0:
                if v is None:
                    # Edge case: Happens if
                    # a keyword is specified in the schema and takes N args
                    # and is also specified as deprecated,
                    # and is specified in the config with 0 arguments
                    # which parses to None for the args
                    continue

                if isinstance(v, ContextString):
                    push_deprecation(
                        schema_info.deprecation_info,
                        ContextList.with_values(token=v.keyword_token, values=[v]),
                    )
                elif isinstance(v, list) and (
                    len(v) == 0 or isinstance(v[0], ContextString)
                ):
                    push_deprecation(schema_info.deprecation_info, v)
                elif isinstance(v[0], list):
                    for arglist in v:
                        push_deprecation(schema_info.deprecation_info, arglist)
        if detected_deprecations:
            for deprecation, line in detected_deprecations:
                ConfigWarning.ert_formatted_warn(
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
        config_dict: ConfigDict,
        filename: str,
    ) -> None:
        """
        check_required checks that all required keywords are in config_dict and
        raises an ConfigValidationError if it is not.
        """
