from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ert import ErtScript
from ert.exceptions import StorageError

if TYPE_CHECKING:
    from ert.storage import Ensemble


class ExportWebvizRFTJob(ErtScript):
    """
    Export RFT observations and simulated responses to a CSV file for Webviz.

    The output file path is "share/results/tables/rft_ert.csv" by default,
    but can be overridden by giving the path as the first parameter:

        EXPORT_WEBVIZ_RFT path/to/output.csv

    The output contains columns for observation data (observation_key, well,
    time, measured_value, std, etc.) joined with simulated response values
    per realization.
    """

    def run(self, ensemble: Ensemble, workflow_args: list[Any]) -> None:
        target_file = (
            "share/results/tables/rft_ert.csv"
            if not workflow_args
            else workflow_args[0]
        )

        # Get RFT observation keys from the experiment
        rft_observations = ensemble.experiment.observations.get("rft")
        if rft_observations is None or rft_observations.is_empty():
            raise StorageError("No RFT observations found in experiment")

        rft_observation_keys = rft_observations["observation_key"].unique().to_list()

        # Get realizations that have responses
        realizations = ensemble.get_realization_list_with_responses()
        if len(realizations) == 0:
            raise StorageError("No realizations with responses found")

        # Fetch observations aligned with responses
        observations_and_responses = ensemble.get_observations_and_responses(
            selected_observations=rft_observation_keys,
            iens_active_index=np.array(realizations),
        )

        # Write to CSV, creating parent directories if needed
        Path(target_file).parent.mkdir(parents=True, exist_ok=True)
        observations_and_responses.write_csv(target_file)
