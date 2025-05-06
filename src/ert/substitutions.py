from __future__ import annotations

import logging
import re
from collections import UserDict
from collections.abc import Mapping
from typing import Any

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

logger = logging.getLogger(__name__)
_PATTERN = re.compile(r"<[^<>]+>")


class Substitutions(UserDict[str, str]):
    def substitute(
        self,
        to_substitute: str,
        context: str = "",
        max_iterations: int = 1000,
        warn_max_iter: bool = True,
    ) -> str:
        """Perform a search-replace on the first argument

        The `context` argument may be used to add information to warnings
        emitted during subsitution.

        """
        return _substitute(self, to_substitute, context, max_iterations, warn_max_iter)

    def substitute_parameters(
        self, to_substitute: str, data: dict[str, dict[str, float | str]]
    ) -> str:
        for values in data.values():
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.6g}"
                else:
                    formatted_value = str(value)
                to_substitute = to_substitute.replace(f"<{key}>", formatted_value)
        return to_substitute

    def substitute_real_iter(
        self, to_substitute: str, realization: int, iteration: int
    ) -> str:
        copy_substituter = self.copy()
        geo_id_key = f"<GEO_ID_{realization}_{iteration}>"
        if geo_id_key in self:
            copy_substituter["<GEO_ID>"] = self[geo_id_key]
        copy_substituter["<IENS>"] = str(realization)
        copy_substituter["<ITER>"] = str(iteration)
        return copy_substituter.substitute(to_substitute)

    def _concise_representation(self) -> str:
        return (
            "[" + ",\n".join([f"({key}, {value})" for key, value in self.items()]) + "]"
        )

    def __repr__(self) -> str:
        return f"<Substitutions({self._concise_representation()})>"

    def __str__(self) -> str:
        return f"Substitutions({self._concise_representation()})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def _serialize(instance: Any, info: Any) -> Any:
            return dict(instance)

        from_str_schema = core_schema.chain_schema(
            [
                core_schema.str_schema(),
                core_schema.no_info_plain_validator_function(cls),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=core_schema.union_schema(
                [
                    from_str_schema,
                    core_schema.is_instance_schema(cls),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize, info_arg=True
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(core_schema.str_schema())


def _substitute(
    substitutions: Mapping[str, str],
    to_substitute: str,
    context: str = "",
    max_iterations: int = 1000,
    warn_max_iter: bool = True,
) -> str:
    """Perform a search-replace on the first argument

    The `context` argument may be used to add information to warnings
    emitted during subsitution.

    """
    substituted_string = to_substitute
    for _ in range(max_iterations):
        substituted_tmp_string = _replace_strings(substitutions, substituted_string)
        if substituted_tmp_string is None:
            break
        substituted_string = substituted_tmp_string
    else:
        if warn_max_iter:
            warning_message = (
                "Reached max iterations while trying to resolve defines in the "
                f"string '{to_substitute}' - after iteratively applying "
                "substitutions given by defines, we ended up with the "
                f"string '{substituted_string}'"
            )
            if context:
                warning_message += f" - context was {context}"
            logger.warning(warning_message)

    return substituted_string


def _replace_strings(substitutions: Mapping[str, str], string: str) -> str | None:
    start = 0
    parts: list[str] = []
    for match in _PATTERN.finditer(string):
        if (val := substitutions.get(match[0])) and val is not None:
            parts.extend((string[start : match.start()], val))
            start = match.end()
    if not parts:
        return None
    parts.append(string[start:])
    return "".join(parts)


def substitute_runpath_name(
    to_substitute: str, realization: int, iteration: int
) -> str:
    """
    To separate between substitution list and what can be substituted in runpath,
    this method is separate from the Substitutions.
    """
    substituted = _substitute(
        {"<IENS>": str(realization), "<ITER>": str(iteration)}, to_substitute
    )

    if "%d" in substituted:
        substituted = substituted % realization  # noqa

    return substituted
