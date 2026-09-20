from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ert import ErtScript

if TYPE_CHECKING:
    from ert.runpaths import Runpaths
    from ert.storage import Ensemble


class ExportRFTJob(ErtScript):
    """
    Export RFT observations and simulated responses to CSV files.

    By default, the output file is "share/results/tables/rft_ert.csv" written to
    each realization's runpath. The filename can be overridden by giving it as
    the first parameter:

        EXPORT_RFT custom_filename.csv

    Each realization gets its own file containing observation data joined with
    simulated response values for that realization. The output is compatible
    with the webviz-subsurface RftPlotter.
    """

    def run(
        self,
        run_paths: Runpaths,
        ensemble: Ensemble,
        workflow_args: list[Any],
    ) -> None:
        filename = (
            workflow_args[0] if workflow_args else "share/results/tables/rft_ert.csv"
        )

        observations_and_responses = ensemble.get_rft_observations_and_responses()

        iteration = ensemble.iteration
        realizations = ensemble.get_realization_list_with_responses()
        paths = run_paths.get_paths(realizations, iteration)

        for realization, runpath in zip(realizations, paths, strict=True):
            realization_data = observations_and_responses.filter(
                observations_and_responses["realization"] == realization
            ).drop("realization")

            target_file = Path(runpath) / filename
            target_file.parent.mkdir(parents=True, exist_ok=True)
            realization_data.write_csv(target_file)
