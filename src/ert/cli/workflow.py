from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ert.job_queue import WorkflowRunner

if TYPE_CHECKING:
    from ert.enkf_main import EnKFMain
    from ert.storage import StorageAccessor


def execute_workflow(
    ert: EnKFMain, storage: StorageAccessor, workflow_name: str
) -> None:
    logger = logging.getLogger(__name__)
    try:
        workflow = ert.ert_config.workflows[workflow_name]
    except KeyError:
        msg = "Workflow {} is not in the list of available workflows"
        logger.error(msg.format(workflow_name))
        return
    runner = WorkflowRunner(workflow, ert, storage)
    runner.run_blocking()
    if not all(v["completed"] for v in runner.workflowReport().values()):
        logger.error(f"Workflow {workflow_name} failed!")
