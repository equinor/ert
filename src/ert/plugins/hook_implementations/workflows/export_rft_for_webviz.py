from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ert import ErtScript

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

        # Fetch observations aligned with responses
        observations_and_responses = ensemble.get_rft_observations_and_responses()

        # Write to CSV, creating parent directories if needed
        Path(target_file).parent.mkdir(parents=True, exist_ok=True)
        observations_and_responses.write_csv(target_file)
