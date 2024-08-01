from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ert.config import ErtConfig, WorkflowJob

from ert.gui.ertnotifier import ErtNotifier
from ert.workflow_runner import WorkflowJobRunner

logger = logging.getLogger(__name__)


class Exporter:
    def __init__(
        self,
        export_job: Optional[WorkflowJob],
        runpath_job: Optional[WorkflowJob],
        notifier: ErtNotifier,
        config: ErtConfig,
    ):
        self.config = config
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
            fixtures={
                "storage": self._notifier.storage,
                "ert_config": self.config,
                "workflow_args": [],
            },
            arguments=[],
        )
        if runpath_job_runner.hasFailed():
            raise UserWarning(f"Failed to execute {self.runpath_job.name}")

        export_job_runner = WorkflowJobRunner(self.export_job)
        user_warn = export_job_runner.run(
            fixtures={"storage": self._notifier.storage, "ert_config": self.config},
            arguments=[
                str(self.config.runpath_file),
                parameters["output_file"],
                parameters["time_index"],
                parameters["column_keys"],
            ],
        )
        if export_job_runner.hasFailed():
            raise UserWarning(f"Failed to execute {self.export_job.name}\n{user_warn}")
