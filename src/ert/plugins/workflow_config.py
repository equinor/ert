from __future__ import annotations

import logging
from argparse import ArgumentParser
from collections.abc import Callable

from ..config.workflow_job import WorkflowJob
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


class ErtScriptWorkflow(WorkflowJob):
    """
    Single workflow configuration object
    """

    def __init__(
        self, ertscript_class: type[ErtScript], name: str | None = None
    ) -> None:
        """
        :param ertscript_class: Class inheriting from ErtScript
        :param name: Optional name for workflow, default is class name
        """
        self.source_package = self._get_source_package(ertscript_class)
        self._description = ertscript_class.__doc__ if ertscript_class.__doc__ else ""
        self._examples: str | None = None
        self._parser: Callable[[], ArgumentParser] | None = None
        self._category = "other"
        super().__init__(
            name=self._get_func_name(ertscript_class, name),
            ert_script=ertscript_class,
            min_args=None,
            max_args=None,
            arg_types=[],
            executable=None,
        )

    @property
    def description(self) -> str:
        """
        A string of valid rst, will be added to the documentation
        """
        return self._description

    @description.setter
    def description(self, description: str) -> None:
        self._description = description

    @property
    def examples(self) -> str | None:
        """
        A string of valid rst, will be added to the documentation
        """
        return self._examples

    @examples.setter
    def examples(self, examples: str | None) -> None:
        self._examples = examples

    @property
    def parser(self) -> Callable[[], ArgumentParser] | None:
        return self._parser

    @parser.setter
    def parser(self, parser: Callable[[], ArgumentParser] | None) -> None:
        self._parser = parser

    @property
    def category(self) -> str:
        """
        A dot separated string
        """
        return self._category

    @category.setter
    def category(self, category: str) -> None:
        self._category = category

    @staticmethod
    def _get_func_name(func: type[ErtScript], name: str | None) -> str:
        return name if name else func.__name__

    @staticmethod
    def _get_source_package(module: type[ErtScript]) -> str:
        base, _, _ = module.__module__.partition(".")
        return base
