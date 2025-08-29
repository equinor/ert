from __future__ import annotations

import logging
import warnings
from argparse import ArgumentParser
from collections.abc import Callable

from ert.config.parsing import SchemaItemType

from ..config.workflow_job import ErtScriptWorkflow
from .ert_script import ErtScript

logger = logging.getLogger(__name__)


class LegacyErtScriptWorkflow:
    """This is a wrapper around ErtScriptWorkflow to keep backwards compatability"""

    def __init__(
        self, actual_workflow: ErtScriptWorkflow, workflow_configs: WorkflowConfigs
    ) -> None:
        self._actual_workflow = actual_workflow
        self._workflow_configs = workflow_configs

    @property
    def name(self) -> str:
        return self._actual_workflow.name

    @name.setter
    def name(self, value: str) -> None:
        self._actual_workflow.name = value

    @property
    def min_args(self) -> int | None:
        return self._actual_workflow.min_args

    @min_args.setter
    def min_args(self, value: int | None) -> None:
        self._actual_workflow.min_args = value

    @property
    def max_args(self) -> int | None:
        return self._actual_workflow.max_args

    @max_args.setter
    def max_args(self, value: int | None) -> None:
        self._actual_workflow.max_args = value

    @property
    def arg_types(self) -> list[SchemaItemType]:
        return self._actual_workflow.arg_types

    @arg_types.setter
    def arg_types(self, value: list[SchemaItemType]) -> None:
        self._actual_workflow.arg_types = value

    @property
    def stop_on_fail(self) -> bool:
        return self._actual_workflow.stop_on_fail

    @stop_on_fail.setter
    def stop_on_fail(self, value: bool) -> None:
        self._actual_workflow.stop_on_fail = value

    @property
    def ert_script(self) -> type[ErtScript]:
        return self._actual_workflow.ert_script

    @ert_script.setter
    def ert_script(self, value: type[ErtScript]) -> None:
        self._actual_workflow.ert_script = value

    @property
    def description(self) -> str:
        return self._actual_workflow.description

    @description.setter
    def description(self, value: str) -> None:
        self._actual_workflow.description = value

    @property
    def category(self) -> str:
        return self._actual_workflow.category

    @category.setter
    def category(self, value: str) -> None:
        self._actual_workflow.category = value

    @property
    def examples(self) -> str | None:
        return self._actual_workflow.examples

    @examples.setter
    def examples(self, value: str | None) -> None:
        self._actual_workflow.examples = value

    @property
    def parser(self) -> Callable[[], ArgumentParser] | None:
        return self._workflow_configs.parsers[self.name]

    @parser.setter
    def parser(self, value: Callable[[], ArgumentParser] | None) -> None:
        self._workflow_configs.parsers[self.name] = value


class WorkflowConfigs:
    """
    Top level workflow config object, holds all workflow configs.
    """

    def __init__(self) -> None:
        self._workflows: list[ErtScriptWorkflow] = []
        self.parsers: dict[str, Callable[[], ArgumentParser] | None] = {}

    def add_workflow(
        self,
        ert_script: type[ErtScript],
        name: str = "",
        description: str = "",
        examples: str | None = None,
        parser: Callable[[], ArgumentParser] | None = None,
        category: str = "other",
    ) -> None:
        """
        :param category: dot separated string
        :param parser: will extract information to use in documentation
        :param examples: must be valid rst, will be added to documentation
        :param description: must be valid rst, defaults to __doc__
        :param ert_script: class which inherits from ErtScript
        :param name: Optional name for workflow (default is name of class)
        """
        workflow = ErtScriptWorkflow(
            name=name or ert_script.__name__,
            ert_script=ert_script,
            description=description,
            examples=examples,
            category=category,
        )
        self._workflows.append(workflow)
        self.parsers[workflow.name] = parser

    def get_workflows(self) -> dict[str, ErtScriptWorkflow]:
        configs = {}
        for workflow in self._workflows:
            if workflow.name in configs:
                logger.info(
                    f"Duplicate workflow name: {workflow.name}, "
                    f"skipping {workflow.ert_script}"
                )
            else:
                configs[workflow.name] = workflow
        return configs


class LegacyWorkflowConfigs(WorkflowConfigs):
    def add_workflow(  # type: ignore
        self,
        ert_script: type[ErtScript],
        name: str = "",
        description: str = "",
        examples: str | None = None,
        parser: Callable[[], ArgumentParser] | None = None,
        category: str = "other",
    ) -> ErtScriptWorkflow:
        """
        :param category: dot separated string
        :param parser: will extract information to use in documentation
        :param examples: must be valid rst, will be added to documentation
        :param description: must be valid rst, defaults to __doc__
        :param ert_script: class which inherits from ErtScript
        :param name: Optional name for workflow (default is name of class)
        :return: Instantiated workflow config.
        """
        warnings.warn(
            "Use of legacy_ertscript_workflow is deprecated. "
            "Please see documentation for how to use ertscript_workflow instead",
            DeprecationWarning,
            stacklevel=1,
        )
        workflow = ErtScriptWorkflow(
            name=name or ert_script.__name__,
            ert_script=ert_script,
            description=description,
            examples=examples,
            category=category,
        )
        self._workflows.append(workflow)
        self.parsers[workflow.name] = parser
        return LegacyErtScriptWorkflow(workflow, self)  # type: ignore
