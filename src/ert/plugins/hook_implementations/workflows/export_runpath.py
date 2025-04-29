from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ert import ErtScript
from ert.runpaths import Runpaths
from ert.validation import rangestring_to_list

if TYPE_CHECKING:
    from ert.storage import Ensemble


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

    def run(
        self, run_paths: Runpaths, ensemble: Ensemble, workflow_args: list[Any]
    ) -> None:
        args = " ".join(workflow_args).split()  # Make sure args is a list of words
        assert ensemble
        iter_ = ensemble.iteration
        reals = ensemble.ensemble_size
        run_paths.write_runpath_list(
            *self.get_ranges(
                args,
                iter_,
                reals,
            )
        )

    def get_ranges(
        self, args: list[str], number_of_iterations: int, number_of_realizations: int
    ) -> tuple[list[int], list[int]]:
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
    def _list_from_rangestring(rangestring: str, size: int) -> list[int]:
        if rangestring == "*":
            return list(range(size))
        else:
            return rangestring_to_list(rangestring)

    def _get_rangestrings(
        self, args: list[str], number_of_realizations: int
    ) -> tuple[str, str]:
        if not args:
            return (
                f"0-{number_of_realizations - 1}",
                "0-0",  # weird default behavior, kept for backwards compatability
            )
        if "|" not in args:
            raise ValueError("Expected | in EXPORT_RUNPATH arguments")
        delimiter = args.index("|")
        return " ".join(args[:delimiter]), " ".join(args[delimiter + 1 :])
