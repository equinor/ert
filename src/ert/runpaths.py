from pathlib import Path
from typing import Iterable, List, Optional, Union

from ert.substitution_list import SubstitutionList


class Runpaths:
    """The Runpaths are the ensemble workspace directories.

    Generally there is one runpath for each realization and iteration, although
    depending on the given format string for the paths they may coincide. There
    is one job name for each of the runpaths.

    :param jobname_format: The format of the job name, e.g., "job_<IENS>"
    :param runpath_format: The format of the runpath, e.g.
        "/path/<case>/ensemble-<IENS>/iteration-<ITER>"
    :param filename: The filename of the runpath list file. Defaults to
        ".ert_runpath_list".
    :param substitute: Function called to perform arbitrary substitution on
        jobname and runpath, e.g., transforms
        "/path/<case>/ensemble-1/iteration-2/" to
        "/path/my_case/ensemble-1/iteration-2". The function is given
        three arguments, the defaults to no substitutions, ie.:

            def default_substitute(to_replace:str, realization:int, iteration:int):
                return to_replace


    """

    def __init__(
        self,
        jobname_format: str,
        runpath_format: str,
        filename: Union[str, Path] = ".ert_runpath_list",
        substitution_list: Optional[SubstitutionList] = None,
        eclbase: Optional[str] = None,
    ):
        self._jobname_format = jobname_format
        self.runpath_list_filename = Path(filename)
        self._runpath_format = str(Path(runpath_format).resolve())
        self._substitution_list = substitution_list or SubstitutionList()
        self._eclbase = eclbase

    def set_ert_ensemble(self, ensemble_name: str) -> None:
        self._substitution_list["<ERT-CASE>"] = ensemble_name
        self._substitution_list["<ERTCASE>"] = ensemble_name

    def get_paths(self, realizations: Iterable[int], iteration: int) -> List[str]:
        return [
            self._substitution_list.substitute_real_iter(
                self._runpath_format, realization, iteration
            )
            for realization in realizations
        ]

    def get_jobnames(self, realizations: Iterable[int], iteration: int) -> List[str]:
        return [
            self._substitution_list.substitute_real_iter(
                self._jobname_format, realization, iteration
            )
            for realization in realizations
        ]

    def write_runpath_list(
        self,
        iteration_numbers: List[int],
        realization_numbers: List[int],
    ) -> None:
        """Writes the runpath_list_file, which lists jobs and runpaths.

        The runpath list file is parsed by some workflows in order to find
        which path was used by each iteration and ensemble.

        Calling write_runpath_list([0,1], [3,4]) with "/cwd/" as the
        current working directory will result in a runpath list file containing:

            003  /cwd/realization-3/iteration-0  job3  000
            004  /cwd/realization-4/iteration-0  job4  000
            003  /cwd/realization-3/iteration-1  job3  001
            004  /cwd/realization-4/iteration-1  job4  001

        The example assumes that jobname_format is "job<IENS>", that there is
        no eclbase and runpath_format is "realization<ITER>/iteration-<IENS>"

        :param iteration_numbers: The list of iterations to write entries for
        :param realization_numbers: The list of realizations to write entries for
        """
        Path(self.runpath_list_filename).parent.mkdir(parents=True, exist_ok=True)
        with open(self.runpath_list_filename, "w", encoding="utf-8") as filehandle:
            for iteration in iteration_numbers:
                for realization in realization_numbers:
                    job_name_or_eclbase = self._substitution_list.substitute_real_iter(
                        self._eclbase if self._eclbase else self._jobname_format,
                        realization,
                        iteration,
                    )
                    runpath = self._substitution_list.substitute_real_iter(
                        self._runpath_format, realization, iteration
                    )

                    filehandle.write(
                        f"{realization:03d}  {runpath}  {job_name_or_eclbase}  {iteration:03d}\n"
                    )
