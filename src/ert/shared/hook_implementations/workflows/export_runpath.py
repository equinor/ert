from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Tuple

from ert.config import ErtScript
from ert.runpaths import Runpaths
from ert.validation import rangestring_to_mask

if TYPE_CHECKING:
    from ert.config import ErtConfig


class ExportRunpathJob(ErtScript):
    """The EXPORT_RUNPATH workflow job writes the runpath file.

    The job can have no arguments, or one can set a range of realizations
    and a range of iterations as arguments. Note: no check is made for whether
    the corresponding runpath has been created.

    Example usage of this job in a workflow:

        EXPORT_RUNPATH

    With no arguments, entries for iterations 0 and all realizations are written
    to the runpath file. The job can also be given ranges of iterations and
    realizations to run:

        EXPORT_RUNPATH 0-5 | *

    "|" is used as a delimiter to separate realizations and iterations. "*" can
    be used to select all realizations or iterations. In the example above,
    entries for realizations 0-5 for all iterations are written to the runpath
    file.
    """

    def run(self, ert_config: ErtConfig, workflow_args: List[Any]) -> None:
        _args = " ".join(workflow_args).split()  # Make sure args is a list of words
        run_paths = Runpaths(
            jobname_format=ert_config.model_config.jobname_format_string,
            runpath_format=ert_config.model_config.runpath_format_string,
            filename=str(ert_config.runpath_file),
            substitution_list=ert_config.substitution_list,
        )
        run_paths.write_runpath_list(
            *self.get_ranges(
                _args,
                ert_config.analysis_config.num_iterations,
                ert_config.model_config.num_realizations,
            )
        )

    def get_ranges(
        self, args: List[str], number_of_iterations: int, number_of_realizations: int
    ) -> Tuple[List[int], List[int]]:
        realizations_rangestring, iterations_rangestring = self._get_rangestrings(
            args, number_of_realizations
        )
        return (
            self._list_from_rangestring(iterations_rangestring, number_of_iterations),
            self._list_from_rangestring(
                realizations_rangestring, number_of_realizations
            ),
        )

    @staticmethod
    def _list_from_rangestring(rangestring: str, size: int) -> List[int]:
        if rangestring == "*":
            return list(range(size))
        else:
            mask = rangestring_to_mask(rangestring, size)
            return [i for i, flag in enumerate(mask) if flag]

    def _get_rangestrings(
        self, args: List[str], number_of_realizations: int
    ) -> Tuple[str, str]:
        if not args:
            return (
                f"0-{number_of_realizations-1}",
                "0-0",  # weird default behavior, kept for backwards compatability
            )
        if "|" not in args:
            raise ValueError("Expected | in EXPORT_RUNPATH arguments")
        delimiter = args.index("|")
        return " ".join(args[:delimiter]), " ".join(args[delimiter + 1 :])
