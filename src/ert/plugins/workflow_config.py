from __future__ import annotations

import logging

from ..config.workflow_job import ErtScriptWorkflow
from .ert_script import ErtScript

logger = logging.getLogger(__name__)


class WorkflowConfigs:
    """
    Top level workflow config object, holds all workflow configs.
    """

    def __init__(self) -> None:
        self._workflows: list[ErtScriptWorkflow] = []

    def add_workflow(
        self, ert_script: type[ErtScript], name: str | None = None
    ) -> ErtScriptWorkflow:
        """

        :param ert_script: class which inherits from ErtScript
        :param name: Optional name for workflow (default is name of class)
        :return: Instantiated workflow config.
        """
        workflow = ErtScriptWorkflow(ert_script, name)
        self._workflows.append(workflow)
        return workflow

    def get_workflows(self) -> dict[str, ErtScriptWorkflow]:
        configs = {}
        for workflow in self._workflows:
            if workflow.name in configs:
                logging.info(
                    f"Duplicate workflow name: {workflow.name}, "
                    f"skipping {workflow.ert_script}"
                )
            else:
                configs[workflow.name] = workflow
        return configs
