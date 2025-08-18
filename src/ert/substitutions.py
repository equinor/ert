from __future__ import annotations

import logging
import re
from collections import UserDict
from collections.abc import Mapping

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

    @staticmethod
    def substitute_parameters(
        to_substitute: str, parameter_values: Mapping[str, Mapping[str, str | float]]
    ) -> str:
        """Applies the substitution '<param_name>' to parameter value
        Args:
            parameter_values: Mapping from parameter name to parameter value
            to_substitute: string to substitute magic strings in
        Returns:
            substituted string
        """
        for values in parameter_values.values():
            for param_name, value in values.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.6g}"
                else:
                    formatted_value = str(value)
                to_substitute = to_substitute.replace(
                    f"<{param_name}>", formatted_value
                )
        return to_substitute

    def substitute_real_iter(
        self, to_substitute: str, realization: int, iteration: int
    ) -> str:
        extra_data = {
            "<IENS>": str(realization),
            "<ITER>": str(iteration),
        }

        geo_id_key = f"<GEO_ID_{realization}_{iteration}>"
        if geo_id_key in self:
            extra_data["<GEO_ID>"] = self[geo_id_key]

        return Substitutions({**self, **extra_data}).substitute(to_substitute)

    def _concise_representation(self) -> str:
        return (
            "[" + ",\n".join([f"({key}, {value})" for key, value in self.items()]) + "]"
        )

    def __repr__(self) -> str:
        return f"<Substitutions({self._concise_representation()})>"

    def __str__(self) -> str:
        return f"Substitutions({self._concise_representation()})"


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
