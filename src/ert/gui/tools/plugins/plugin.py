from typing import TYPE_CHECKING

from ert._c_wrappers.job_queue import ErtScript

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ert._c_wrappers.enkf import EnKFMain
    from ert._c_wrappers.job_queue import ErtPlugin, WorkflowJob


class Plugin:
    def __init__(self, ert: "EnKFMain", workflow_job: "WorkflowJob"):
        """
        @type workflow_job: WorkflowJob
        """
        self.__ert = ert
        self.__workflow_job = workflow_job
        self.__parent_window = None

        script = self.__loadPlugin()
        self.__name = script.getName()
        self.__description = script.getDescription()

    def __loadPlugin(self) -> "ErtPlugin":
        script_obj = ErtScript.loadScriptFromFile(self.__workflow_job.script)
        script = script_obj(self.__ert)
        return script

    def getName(self) -> str:
        return self.__name

    def getDescription(self) -> str:
        return self.__description

    def getArguments(self):
        """
         Returns a list of arguments. Either from GUI or from arbitrary code.
         If the user for example cancels in the GUI a CancelPluginException is raised.
        @rtype: list"""
        script = self.__loadPlugin()
        return script.getArguments(self.__parent_window)

    def setParentWindow(self, parent_window):
        self.__parent_window = parent_window

    def getParentWindow(self) -> "QWidget":
        return self.__parent_window

    def ert(self) -> "EnKFMain":
        return self.__ert

    def getWorkflowJob(self) -> "WorkflowJob":
        return self.__workflow_job
