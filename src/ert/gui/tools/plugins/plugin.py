from typing import TYPE_CHECKING, Any, List

from ert import ErtScript

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ert.config import ErtPlugin, WorkflowJob
    from ert.enkf_main import EnKFMain
    from ert.gui.ertnotifier import ErtNotifier


class Plugin:
    def __init__(
        self, ert: "EnKFMain", notifier: "ErtNotifier", workflow_job: "WorkflowJob"
    ):
        self.__ert = ert
        self.__notifier = notifier
        self.__workflow_job = workflow_job
        self.__parent_window = None

        script = self.__loadPlugin()
        self.__name = script.getName()
        self.__description = script.getDescription()

    def __loadPlugin(self) -> "ErtPlugin":
        script_obj = ErtScript.loadScriptFromFile(self.__workflow_job.script)
        script = script_obj(
            self.__ert, self.__notifier._storage, ensemble=self.__notifier.current_case
        )
        return script

    def getName(self) -> str:
        return self.__name

    def getDescription(self) -> str:
        return self.__description

    def getArguments(self) -> List[Any]:
        """
        Returns a list of arguments. Either from GUI or from arbitrary code.
        If the user for example cancels in the GUI a CancelPluginException is raised.
        """
        script = self.__loadPlugin()
        return script.getArguments(self.__parent_window)

    def setParentWindow(self, parent_window):
        self.__parent_window = parent_window

    def getParentWindow(self) -> "QWidget":
        return self.__parent_window

    def ert(self) -> "EnKFMain":
        return self.__ert

    @property
    def storage(self):
        return self.__notifier.storage

    @property
    def ensemble(self):
        return self.__notifier.current_case

    def getWorkflowJob(self) -> "WorkflowJob":
        return self.__workflow_job
