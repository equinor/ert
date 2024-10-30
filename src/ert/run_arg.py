from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

from ert.runpaths import Runpaths

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from ert.storage import Ensemble


@dataclass
class RunArg:
    run_id: str
    ensemble_storage: Ensemble
    iens: int
    itr: int
    runpath: str
    job_name: str
    active: bool = True
    runpaths: Optional[Runpaths] = None
    # Below here is legacy related to Everest
    queue_index: Optional[int] = None

    def __post_init__(self) -> None:
        if self.runpaths is None:
            self.runpaths = Runpaths(self.job_name, self.runpath)

    def file_in_runpath(self, filename: str) -> str:
        assert self.runpaths is not None
        if "%d" in filename:
            filename = filename % self.iens  # noqa
        return self.runpaths.runpath_file(
            filename, realization=self.iens, iteration=self.itr
        )


def create_run_arguments(
    runpaths: Runpaths,
    active_realizations: Union[List[bool], npt.NDArray[np.bool_]],
    ensemble: Ensemble,
) -> List[RunArg]:
    iteration = ensemble.iteration
    run_args = []
    runpaths.set_ert_ensemble(ensemble.name)
    paths = runpaths.get_paths(range(len(active_realizations)), iteration)
    job_names = runpaths.get_jobnames(range(len(active_realizations)), iteration)

    for iens, (run_path, job_name, active) in enumerate(
        zip(paths, job_names, active_realizations)
    ):
        run_args.append(
            RunArg(
                str(ensemble.id),
                ensemble,
                iens,
                iteration,
                run_path,
                job_name,
                runpaths=runpaths,
                active=active,
            )
        )
    return run_args
