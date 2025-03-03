from __future__ import annotations

import logging
from argparse import ArgumentParser
from collections.abc import Callable

from ..config.ert_script import ErtScript
from ..config.workflow_job import ErtScriptWorkflow

logger = logging.getLogger(__name__)


class WorkflowConfigs:
    """
    Top level workflow config object, holds all workflow configs.
    """

    def __init__(self) -> None:
        self._workflows: list[ErtScriptWorkflow] = []

    def add_workflow(
        self,
        ert_script: type[ErtScript],
        name: str = "",
        description: str = "",
        examples: str | None = None,
        parser: Callable[[], ArgumentParser] | None = None,
        category: str = "other",
    ) -> ErtScriptWorkflow:
        """
        :param category:
        :param parser:
        :param examples:
        :param description:
        :param ert_script: class which inherits from ErtScript
        :param name: Optional name for workflow (default is name of class)
        :return: Instantiated workflow config.
        """
        workflow = ErtScriptWorkflow(
            name=name if name else ert_script.__name__,
            ert_script=ert_script,
            description=description,
            examples=examples,
            parser=parser,
            category=category,
        )
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
