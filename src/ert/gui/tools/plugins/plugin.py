from typing import TYPE_CHECKING, Any

from ert._c_wrappers.job_queue import ErtScript

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ert._c_wrappers.enkf import EnKFMain
    from ert._c_wrappers.job_queue import ErtPlugin, WorkflowJob
    from ert._c_wrappers.job_queue.run_status import RunStatus


class Plugin:
    def __init__(self, ert: "EnKFMain", workflow_job: "WorkflowJob"):
        """
        @type workflow_job: WorkflowJob
        """
        self.__ert = ert
        self.__workflow_job = workflow_job
        self.__parent_window = None

        self.loaded_script = self.__loadPlugin()
        self.__name = self.loaded_script.getName()
        self.__description = self.loaded_script.getDescription()

    def __loadPlugin(self) -> "ErtPlugin":
        script_obj = ErtScript.loadScriptFromFile(self.__workflow_job.script)
        return script_obj(self.__ert)

    def getName(self) -> str:
        return self.__name

    def getDescription(self) -> str:
        return self.__description

    def getArguments(self):
        """
         Returns a list of arguments. Either from GUI or from arbitrary code.
         If the user for example cancels in the GUI a CancelPluginException is raised.
        @rtype: list"""
        return self.loaded_script.getArguments(self.__parent_window)

    def setParentWindow(self, parent_window):
        self.__parent_window = parent_window

    def getParentWindow(self) -> "QWidget":
        return self.__parent_window

    def cancel(self):
        self.__workflow_job.cancel()

    def run(self, arguments) -> Any:
        return self.__workflow_job.run(self.__ert, arguments)

    @property
    def run_status(self) -> "RunStatus":
        return self.__workflow_job.run_status
