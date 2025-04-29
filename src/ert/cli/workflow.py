from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ert.runpaths import Runpaths
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

    runner = WorkflowRunner(
        workflow=workflow,
        fixtures={
            "storage": storage,
            "random_seed": ert_config.random_seed,
            "reports_dir": str(ert_config.analysis_config.log_path),
            "observation_settings": ert_config.analysis_config.observation_settings,
            "es_settings": ert_config.analysis_config.es_settings,
            "run_paths": Runpaths(
                jobname_format=ert_config.runpath_config.jobname_format_string,
                runpath_format=ert_config.runpath_config.runpath_format_string,
                filename=str(ert_config.runpath_file),
                substitutions=ert_config.substitutions,
                eclbase=ert_config.runpath_config.eclbase_format_string,
            ),
            "ensemble": None,
        },
    )
    runner.run_blocking()
    if not all(v["completed"] for v in runner.workflowReport().values()):
        logger.error(f"Workflow {workflow_name} failed!")
