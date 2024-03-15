from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)
_PATTERN = re.compile("<[^<>]+>")


# Python 3.8 does not implement UserDict as a MutableMapping, meaning it's not
# possible to specify the key and value types.
#
# Instead, we only set the types during type checking
if TYPE_CHECKING:
    from collections import UserDict

    _UserDict = UserDict[str, str]
else:
    from collections import UserDict as _UserDict


class SubstitutionList(_UserDict):
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
        substituted_string = to_substitute
        for _ in range(max_iterations):
            substituted_tmp_string = _replace_strings(self, substituted_string)
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
        return f"<SubstitutionList({self._concise_representation()})>"

    def __str__(self) -> str:
        return f"SubstitutionList({self._concise_representation()})"


def _replace_strings(subst_list: SubstitutionList, string: str) -> Optional[str]:
    start = 0
    parts = []
    for match in _PATTERN.finditer(string):
        if (val := subst_list.get(match[0])) and val is not None:
            parts.append(string[start : match.start()])
            parts.append(val)
            start = match.end()
    if not parts:
        return None
    parts.append(string[start:])
    return "".join(parts)
