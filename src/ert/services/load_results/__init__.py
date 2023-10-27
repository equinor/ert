from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np

from ert.config.ext_job import ExtJob
from ert.config.queue_system import QueueSystem
from ert.realization_state import RealizationState

if TYPE_CHECKING:
    import numpy.typing as npt

    from ert.storage import EnsembleReader


# SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "__main__.py")
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "job.sh")


def load_results(
    ensemble: EnsembleReader,
    runpath_format: str,
    iteration: int,
    realizations: npt.NDArray[np.bool_],
) -> int:
    from .__main__ import load_results as fun

    success = 0
    for iens in np.flatnonzero(realizations):
        runpath = runpath_format.replace("<IENS>", str(iens)).replace(
            "<ITER>", str(iteration)
        )

        if fun(
            ensemble.get_realization(int(iens), mode="w"),
            os.path.abspath(runpath),
            ensemble.experiment.parameter_configuration.values(),
            ensemble.experiment.response_configuration.values(),
        ):
            success += 1
            ensemble.state_map[iens] = RealizationState.HAS_DATA

    return success


def load_results_job(queue_system: QueueSystem) -> ExtJob:
    return ExtJob(
        name="Load Results",
        executable=SCRIPT_PATH,
        stdout_file="ert_load_results.stdout",
        stderr_file="ert_load_results.stderr",
        arglist=[
            sys.executable,
            "--storage-path",
            "<STORAGE-PATH>",
            "--ensemble",
            "<ENSEMBLE-ID>",
            "--index",
            "<IENS>",
        ],
        environment={"_ERT_QUEUE": str(queue_system)},
    )
