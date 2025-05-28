from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ert.config import ErtConfig, _WorkflowJob

from ert.gui.ertnotifier import ErtNotifier
from ert.workflow_runner import WorkflowJobRunner

logger = logging.getLogger(__name__)


class Exporter:
    def __init__(
        self,
        export_job: _WorkflowJob | None,
        notifier: ErtNotifier,
        config: ErtConfig,
    ) -> None:
        self.config = config
        self.export_job = export_job
        self._notifier = notifier

    def run_export(self, parameters: list[Any]) -> None:
        if self.export_job is None:
            raise UserWarning("Could not find export_job job")

        export_job_runner = WorkflowJobRunner(self.export_job)
        user_warn = export_job_runner.run(
            fixtures={"storage": self._notifier.storage},
            arguments=parameters,
        )
        if export_job_runner.hasFailed():
            raise UserWarning(f"Failed to execute {self.export_job.name}\n{user_warn}")
