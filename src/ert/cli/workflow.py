from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ert.workflow_runner import WorkflowRunner

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.storage import Storage


def execute_workflow(
    ert_config: ErtConfig, storage: Storage, workflow_name: str
) -> None:
    logger = logging.getLogger(__name__)
    try:
        workflow = ert_config.workflows[workflow_name]
    except KeyError:
        msg = "Workflow {} is not in the list of available workflows"
        logger.error(msg.format(workflow_name))
        return
    runner = WorkflowRunner(workflow, storage, ert_config=ert_config)
    runner.run_blocking()
    if not all(v["completed"] for v in runner.workflowReport().values()):
        logger.error(f"Workflow {workflow_name} failed!")
