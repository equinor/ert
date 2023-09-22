import time
from functools import partial
from threading import Thread
from typing import TYPE_CHECKING

from ert.config import CancelPluginException
from ert.job_queue import WorkflowJobRunner

from .process_job_dialog import ProcessJobDialog

if TYPE_CHECKING:
    from .plugin import Plugin


class PluginRunner:
    def __init__(self, plugin: "Plugin"):
        super().__init__()

        self.__plugin = plugin

        self.__plugin_finished_callback = lambda: None

        self.__result = None
        self._runner = WorkflowJobRunner(plugin.getWorkflowJob())
        self.poll_thread = None

    def run(self):
        try:
            plugin = self.__plugin

            arguments = plugin.getArguments()
            dialog = ProcessJobDialog(plugin.getName(), plugin.getParentWindow())
            dialog.setObjectName("process_job_dialog")

            dialog.cancelConfirmed.connect(self.cancel)

            run_function = partial(self.__runWorkflowJob, plugin, arguments)

            workflow_job_thread = Thread(name="ert_gui_workflow_job_thread")
            workflow_job_thread.daemon = True
            workflow_job_thread.run = run_function
            workflow_job_thread.start()

            poll_function = partial(self.__pollRunner, dialog)

            self.poll_thread = Thread(name="ert_gui_workflow_job_poll_thread")
            self.poll_thread.daemon = True
            self.poll_thread.run = poll_function
            self.poll_thread.start()

            dialog.show()
        except CancelPluginException:
            print("Plugin cancelled before execution!")

    def __runWorkflowJob(self, plugin, arguments):
        self.__result = self._runner.run(
            plugin.ert(), plugin.storage, plugin.ensemble, arguments
        )

    def __pollRunner(self, dialog):
        self.wait()

        details = ""
        if self.__result is not None:
            details = str(self.__result)

        if self._runner.hasFailed():
            dialog.presentError.emit(
                "Job failed!",
                f"The job '{self.__plugin.getName()}' has failed while running!",
                details,
            )
            dialog.disposeDialog.emit()
        elif self._runner.isCancelled():
            dialog.presentInformation.emit(
                "Job cancelled!",
                f"The job '{self.__plugin.getName()}' was cancelled successfully!",
                details,
            )
            dialog.disposeDialog.emit()
        else:
            dialog.presentInformation.emit(
                "Job completed!",
                f"The job '{self.__plugin.getName()}' was completed successfully!",
                details,
            )
            dialog.disposeDialog.emit()

        self.__plugin_finished_callback()

    def isRunning(self) -> bool:
        return self._runner.isRunning()

    def isCancelled(self) -> bool:
        return self._runner.isCancelled()

    def cancel(self):
        if self.isRunning():
            self._runner.cancel()

    def wait(self):
        while self.isRunning():
            time.sleep(1)

    def setPluginFinishedCallback(self, callback):
        self.__plugin_finished_callback = callback
