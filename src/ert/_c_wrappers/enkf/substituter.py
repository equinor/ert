from collections import defaultdict
from typing import Dict, Tuple


class Substituter:
    """Performs substitution of strings per realization and iteration.

    The Substituter manages some special substitution strings:
        <ITER>: The iteration number, provided when substitution is performed.
        <IENS>: The realization number, provided when substitution is performed.
    """

    def __init__(self, global_substitutions: Dict[str, str] = None):
        """
        :param global_substitutions: List of substitutions that should be
            performed, in the same way regardless of realization and iteration, ie.

                >>> substituter = Substituter([("<case_name>", "my_case")])
                >>> substituter.substitute("the case is: <case_name>", 0, 0)
                "the case is: my_case"
        """
        if global_substitutions is None:
            self._global_substitutions = {}
        else:
            self._global_substitutions = global_substitutions
        self._local_substitutions: Dict[Tuple[int, int], Dict[str, str]] = defaultdict(
            dict
        )

    def add_substitution(self, key: str, value: str, realization: int, iteration: int):
        """Adds a keyword to be substituted an the given realization/iteration.

        example:
            >>> substituter = Substituter()
            >>> substituter.add_substitution("<GEO_ID>", "my_geo_id", 0, 0)
            >>> substituter.substitute("the geo id is: <GEO_ID>", 0, 0)
            "the geo id is: my_geo_id"
            >>> substituter.substitute("the geo id is: <GEO_ID>", 9, 9)
            "the geo id is: <GEO_ID>"
        """
        self._local_substitutions[(realization, iteration)][key] = str(value)

    def add_global_substitution(self, key: str, value: str):
        """Sets the <GEO_ID> substitution string for the given realization and iteration

        example:
            >>> substituter = Substituter()
            >>> substituter.add_global_substitution("<RUNPATH_FILE>", "/path/filename")
            >>> substituter.substitute("the file is: <RUNPATH_FILE>", 0, 0)
            "the file is: /path/filename"
            >>> substituter.substitute("the file is: <RUNPATH_FILE>", 9, 8)
            "the file is: /path/filename"
        """
        self._global_substitutions[key] = str(value)

    def get_substitutions(self, realization: int, iteration: int) -> Dict[str, str]:
        """

        Example:

            >>> substituter = Substituter([("<GLOBAL>", "global")])
            >>> substituter.get_substitutions(0, 1)
            {"<GLOBAL>": "global", "<IENS>": 0, "<ITER>": 1}

        :return: All substitutions (global and local) to be applied for
            the given realization and iteration.
        """
        result = self._global_substitutions.copy()
        result.update(**self._local_substitutions[(realization, iteration)])
        result["<IENS>"] = str(realization)
        result["<ITER>"] = str(iteration)
        return result

    def substitute(self, to_substitute: str, realization: int, iteration: int) -> str:
        for key, value in self.get_substitutions(realization, iteration).items():
            to_substitute = to_substitute.replace(key, value)
        return to_substitute
