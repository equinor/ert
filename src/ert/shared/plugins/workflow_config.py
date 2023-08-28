from __future__ import annotations

import inspect
import logging
import os
import tempfile
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Optional, Type


class WorkflowConfigs:
    """
    Top level workflow config object, holds all workflow configs.
    """

    def __init__(self) -> None:
        self._temp_dir = tempfile.mkdtemp()
        self._workflows: List[WorkflowConfig] = []

    def add_workflow(
        self, ert_script: Type[Any], name: Optional[str] = None
    ) -> WorkflowConfig:
        """

        :param ert_script: class which inherits from ErtScript
        :param name: Optional name for workflow (default is name of class)
        :return: Instantiated workflow config.
        :type: :class:`ert.shared.plugins.workflow_config.WorkflowConfig`
        """
        workflow = WorkflowConfig(ert_script, self._temp_dir, name)
        self._workflows.append(workflow)
        return workflow

    def get_workflows(self) -> Dict[str, str]:
        configs = {}
        for workflow in self._workflows:
            if workflow.name in configs:
                logging.info(
                    f"Duplicate workflow name: {workflow.name}, "
                    f"skipping {workflow.function_dir}"
                )
            else:
                configs[workflow.name] = workflow.config_path
        return configs


class WorkflowConfig:
    """
    Single workflow configuration object

    """

    def __init__(
        self, ertscript_class: Type[Any], tmpdir: str, name: Optional[str] = None
    ) -> None:
        """
        :param ertscript_class: Class inheriting from ErtScript
        :param tmpdir: Where workflow config is generated
        :param name: Optional name for workflow, default is class name
        """
        self.func = ertscript_class
        self.name = self._get_func_name(ertscript_class, name)
        self.function_dir = os.path.abspath(inspect.getfile(ertscript_class))
        self.source_package = self._get_source_package(self.func)
        self.config_path = self._write_workflow_config(tmpdir)
        self._description = ertscript_class.__doc__ if ertscript_class.__doc__ else ""
        self._examples: Optional[str] = None
        self._parser: Optional[Callable[[], ArgumentParser]] = None
        self._category = "other"

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
    def examples(self) -> Optional[str]:
        """
        A string of valid rst, will be added to the documentation
        """
        return self._examples

    @examples.setter
    def examples(self, examples: Optional[str]) -> None:
        self._examples = examples

    @property
    def parser(self) -> Optional[Callable[[], ArgumentParser]]:
        return self._parser

    @parser.setter
    def parser(self, parser: Optional[Callable[[], ArgumentParser]]) -> None:
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
    def _get_func_name(func: Type[Any], name: Optional[str]) -> str:
        return name if name else func.__name__

    def _write_workflow_config(self, output_dir: str) -> str:
        file_path = os.path.join(output_dir, self.name.upper())
        with open(file_path, "w", encoding="utf-8") as f_out:
            f_out.write("INTERNAL      True\n")
            f_out.write(f"SCRIPT        {self.function_dir}")
        return file_path

    @staticmethod
    def _get_source_package(module: Type[Any]) -> str:
        base, _, _ = module.__module__.partition(".")
        return base
