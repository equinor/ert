from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ert import ErtScript

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ert.config import ErtPlugin, WorkflowJob
    from ert.gui.ertnotifier import ErtNotifier
    from ert.storage import Ensemble, LocalStorage


class Plugin:
    def __init__(self, notifier: ErtNotifier, workflow_job: WorkflowJob):
        self.__notifier = notifier
        self.__workflow_job = workflow_job
        self.__parent_window: Optional[QWidget] = None

        script = self.__loadPlugin()
        self.__name = script.getName()
        self.__description = script.getDescription()

    def __loadPlugin(self) -> ErtPlugin:
        script_obj = ErtScript.loadScriptFromFile(self.__workflow_job.script)  # type: ignore
        script = script_obj()
        return script  # type: ignore

    def getName(self) -> str:
        return self.__name

    def getDescription(self) -> str:
        return self.__description

    def getArguments(self, fixtures: Dict[str, Any]) -> List[Any]:
        """
        Returns a list of arguments. Either from GUI or from arbitrary code.
        If the user for example cancels in the GUI a CancelPluginException is raised.
        """
        script = self.__loadPlugin()
        fixtures["parent"] = self.__parent_window
        func_args = inspect.signature(script.getArguments).parameters
        arguments = script.insert_fixtures(func_args, fixtures)

        # Part of deprecation
        script._ert = fixtures.get("ert_config")
        script._ensemble = fixtures.get("ensemble")
        script._storage = fixtures.get("storage")
        return script.getArguments(*arguments)

    def setParentWindow(self, parent_window: Optional[QWidget]) -> None:
        self.__parent_window = parent_window

    def getParentWindow(self) -> Optional[QWidget]:
        return self.__parent_window

    def ert(self) -> None:
        raise NotImplementedError("No such property")

    @property
    def storage(self) -> LocalStorage:
        return self.__notifier.storage

    @property
    def ensemble(self) -> Optional[Ensemble]:
        return self.__notifier.current_ensemble

    def getWorkflowJob(self) -> WorkflowJob:
        return self.__workflow_job
