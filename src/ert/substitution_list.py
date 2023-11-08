from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Optional, Tuple, no_type_check

from resdata.rd_util import get_num_cpu as get_num_cpu_from_data_file

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
    @no_type_check
    @staticmethod
    def from_dict(config_dict) -> SubstitutionList:
        subst_list = SubstitutionList()

        for key, val in config_dict.get("DEFINE", []):
            subst_list[key] = val

        if "<CONFIG_PATH>" not in subst_list:
            subst_list["<CONFIG_PATH>"] = config_dict.get(
                "CONFIG_DIRECTORY", os.getcwd()
            )

        num_cpus = config_dict.get("NUM_CPU")
        if num_cpus is None and "DATA_FILE" in config_dict:
            num_cpus = get_num_cpu_from_data_file(config_dict.get("DATA_FILE"))
        if num_cpus is None:
            num_cpus = 1
        subst_list["<NUM_CPU>"] = str(num_cpus)

        for key, val in config_dict.get("DATA_KW", []):
            subst_list[key] = val

        return subst_list

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


def _split_by(string: str, delim: str) -> Tuple[str, str]:
    """Find substring in string while ignoring quoted strings.

    Note that escape characters are not allowed.
    """
    assert len(delim) == 1, "delimiter must be of size 1"
    quote_char: Optional[str] = None

    for index, char in enumerate(string):
        # End of quotation
        if quote_char == char:
            quote_char = None

        # Inside of a quotation
        elif quote_char is not None:
            pass

        # Start of quatation
        elif char in ("'", '"'):
            quote_char = char

        # Outside of quotation
        elif char == delim:
            return string[:index].strip(), string[index + 1 :].strip()
    return string, ""
