from __future__ import annotations

import contextlib
import logging
import sys
from typing import TYPE_CHECKING

from ert.runpaths import Runpaths
from ert.workflow_runner import (
    WorkflowJobFailedError,
    WorkflowJobStatus,
    WorkflowRunner,
)

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

    runner = WorkflowRunner(
        workflow=workflow,
        fixtures={
            "storage": storage,
            "random_seed": ert_config.random_seed,
            "reports_dir": str(ert_config.analysis_config.log_path),
            "observation_settings": ert_config.analysis_config.observation_settings,
            "es_settings": ert_config.analysis_config.es_settings,
            "run_paths": Runpaths.from_config(ert_config),
            "ensemble": None,
        },
    )
    with contextlib.suppress(WorkflowJobFailedError):
        runner.run_blocking()

    for result in runner.workflow_job_results():
        if result.status is WorkflowJobStatus.FAILED:
            print(f"Workflow job {result.name} failed: {result.error}", file=sys.stderr)

    if not all(v["completed"] for v in runner.workflowReport().values()):
        logger.error(f"Workflow {workflow_name} failed!")
