from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd

from ert.analysis.event import DataSection

if TYPE_CHECKING:
    from ert.config import WorkflowJob

from ert.gui.ertnotifier import ErtNotifier
from ert.workflow_runner import WorkflowJobRunner

logger = logging.getLogger(__name__)


class Exporter:
    def __init__(
        self,
        export_job: Optional[WorkflowJob],
        runpath_job: Optional[WorkflowJob],
        notifier: ErtNotifier,
        runpath_file: Path,
    ):
        self.runpath_file = runpath_file
        self.export_job = export_job
        self.runpath_job = runpath_job
        self._notifier = notifier

    def is_valid(self) -> bool:
        if self.export_job is None:
            logger.error("Export not available because export_job is not installed.")
            return False

        if self.runpath_job is None:
            logger.error("Export not available because runpath_job is not installed.")
            return False

        return True

    def run_export(self, parameters: Dict[str, Any]) -> None:
        if self.export_job is None:
            raise UserWarning("Could not find export_job job")
        if self.runpath_job is None:
            raise UserWarning("Could not find runpath_job job")

        runpath_job_runner = WorkflowJobRunner(self.runpath_job)

        runpath_job_runner.run(
            fixtures={"storage": self._notifier.storage},
            arguments=[],
        )
        if runpath_job_runner.hasFailed():
            raise UserWarning(f"Failed to execute {self.runpath_job.name}")

        export_job_runner = WorkflowJobRunner(self.export_job)
        user_warn = export_job_runner.run(
            fixtures={"storage": self._notifier.storage},
            arguments=[
                str(self.runpath_file),
                parameters["output_file"],
                parameters["time_index"],
                parameters["column_keys"],
            ],
        )
        if export_job_runner.hasFailed():
            raise UserWarning(f"Failed to execute {self.export_job.name}\n{user_warn}")


def csv_event_to_report(name: str, data: DataSection, output_path: Path) -> None:
    fname = str(name).strip().replace(" ", "_")
    fname = re.sub(r"(?u)[^-\w]", "", fname)
    f_path = output_path / fname
    f_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data.data, columns=data.header)
    with open(f_path.with_suffix(".report"), "w", encoding="utf-8") as fout:
        if data.extra:
            for k, v in data.extra.items():
                fout.write(f"{k}: {v}\n")
        fout.write(df.to_markdown(tablefmt="simple_outline", floatfmt=".4f"))
    df.to_csv(f_path.with_suffix(".csv"))
