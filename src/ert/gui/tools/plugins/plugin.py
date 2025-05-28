from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from ert.config.workflow_job import ErtScriptWorkflow
from ert.plugins import ErtPlugin, WorkflowFixtures

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

    from ert.config import _WorkflowJob
    from ert.gui.ertnotifier import ErtNotifier
    from ert.storage import Ensemble, LocalStorage


class Plugin:
    def __init__(self, notifier: ErtNotifier, workflow_job: ErtScriptWorkflow) -> None:
        self.__notifier = notifier
        self.__workflow_job = workflow_job
        self.__parent_window: QWidget | None = None

        script = self.__loadPlugin()
        self.__name = script.getName()
        self.__description = script.getDescription()

    def __loadPlugin(self) -> ErtPlugin:
        script_obj = self.__workflow_job.ert_script
        script = script_obj()
        assert isinstance(script, ErtPlugin)
        return script

    def getName(self) -> str:
        return self.__name

    def getDescription(self) -> str:
        return self.__description

    def getArguments(self, fixtures: WorkflowFixtures) -> list[Any]:
        """
        Returns a list of arguments. Either from GUI or from arbitrary code.
        If the user for example cancels in the GUI a CancelPluginException is raised.
        """
        script = self.__loadPlugin()
        fixtures["parent"] = self.__parent_window
        func_args = inspect.signature(script.getArguments).parameters
        arguments = script.insert_fixtures(func_args, fixtures)

        return script.getArguments(*arguments)

    def setParentWindow(self, parent_window: QWidget | None) -> None:
        self.__parent_window = parent_window

    def getParentWindow(self) -> QWidget | None:
        return self.__parent_window

    def ert(self) -> None:
        raise NotImplementedError("No such property")

    @property
    def storage(self) -> LocalStorage:
        return self.__notifier.storage

    @property
    def ensemble(self) -> Ensemble | None:
        return self.__notifier.current_ensemble

    def getWorkflowJob(self) -> _WorkflowJob:
        return self.__workflow_job
