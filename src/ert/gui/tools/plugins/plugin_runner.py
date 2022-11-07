import time
from functools import partial
from threading import Thread

from ert._c_wrappers.job_queue.ert_plugin import CancelPluginException

from .process_job_dialog import ProcessJobDialog


class PluginRunner:
    def __init__(self, plugin):
        """
        @type plugin: Plugin
        """
        super().__init__()

        self.__plugin = plugin

        self.__plugin_finished_callback = lambda: None

        self.__result = None

    def run(self):
        try:
            plugin = self.__plugin

            arguments = plugin.getArguments()
            dialog = ProcessJobDialog(plugin.getName(), plugin.getParentWindow())

            dialog.cancelConfirmed.connect(self.cancel)

            run_function = partial(self.__runWorkflowJob, plugin, arguments)

            workflow_job_thread = Thread(name="ert_gui_workflow_job_thread")
            workflow_job_thread.daemon = True
            workflow_job_thread.run = run_function
            workflow_job_thread.start()

            poll_function = partial(self.__pollRunner, plugin, dialog)

            poll_thread = Thread(name="ert_gui_workflow_job_poll_thread")
            poll_thread.daemon = True
            poll_thread.run = poll_function
            poll_thread.start()

            dialog.show()
        except CancelPluginException:
            print("Plugin cancelled before execution!")

    def __runWorkflowJob(self, plugin, arguments):
        workflow_job = plugin.getWorkflowJob()
        self.__result = workflow_job.run(plugin.ert(), arguments)

    def __pollRunner(self, plugin, dialog):
        self.wait()

        details = ""
        if self.__result is not None:
            details = str(self.__result)

        if plugin.getWorkflowJob().hasFailed():
            dialog.presentError.emit(
                "Job failed!",
                f"The job '{plugin.getName()}' has failed while running!",
                details,
            )
            dialog.disposeDialog.emit()
        elif plugin.getWorkflowJob().isCancelled():
            dialog.presentInformation.emit(
                "Job cancelled!",
                f"The job '{plugin.getName()}' was cancelled successfully!",
                details,
            )
            dialog.disposeDialog.emit()
        else:
            dialog.presentInformation.emit(
                "Job completed!",
                f"The job '{plugin.getName()}' was completed successfully!",
                details,
            )
            dialog.disposeDialog.emit()

        self.__plugin_finished_callback()

    def isRunning(self) -> bool:
        return self.__plugin.getWorkflowJob().isRunning()

    def isCancelled(self) -> bool:
        return self.__plugin.getWorkflowJob().isCancelled()

    def cancel(self):
        if self.isRunning():
            self.__plugin.getWorkflowJob().cancel()

    def wait(self):
        while self.isRunning():
            time.sleep(1)

    def setPluginFinishedCallback(self, callback):
        self.__plugin_finished_callback = callback
