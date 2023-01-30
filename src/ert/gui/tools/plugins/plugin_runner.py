import time
from functools import partial
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable

from ert._c_wrappers.job_queue.ert_plugin import CancelPluginException

from .process_job_dialog import ProcessJobDialog

if TYPE_CHECKING:
    from .plugin import Plugin


class PluginRunner:
    def __init__(
        self, plugin: "Plugin", on_finish_callback: Callable[..., Any] = lambda: None
    ):
        """
        @type plugin: Plugin
        """
        super().__init__()

        self.__plugin = plugin
        self.__plugin_finished_callback = on_finish_callback

    def run(self):
        try:
            plugin = self.__plugin

            dialog = ProcessJobDialog(plugin.getName(), plugin.getParentWindow())
            dialog.cancelConfirmed.connect(self.cancel)

            workflow_job_thread = Thread(name="ert_gui_workflow_job_thread")
            workflow_job_thread.daemon = True
            workflow_job_thread.run = partial(plugin.run, plugin.getArguments())
            workflow_job_thread.start()

            poll_function = partial(self.__pollRunner, plugin, dialog)

            poll_thread = Thread(name="ert_gui_workflow_job_poll_thread")
            poll_thread.daemon = True
            poll_thread.run = poll_function
            poll_thread.start()

            dialog.show()
        except CancelPluginException:
            print("Plugin cancelled before execution!")

    def __pollRunner(self, plugin: "Plugin", dialog):
        self.wait()
        status = plugin.run_status
        if status.has_failed():
            dialog.presentError.emit(
                "Job failed!",
                f"The job '{plugin.getName()}' has failed while running!",
                status.stderrdata,
            )
        elif status.was_canceled():
            dialog.presentInformation.emit(
                "Job cancelled!",
                f"The job '{plugin.getName()}' was cancelled successfully!",
                status.stdoutdata,
            )
        elif status.has_finished():
            dialog.presentInformation.emit(
                "Job completed!",
                f"The job '{plugin.getName()}' was completed successfully!",
                status.stdoutdata,
            )

        dialog.disposeDialog.emit()
        self.__plugin_finished_callback()

    def cancel(self):
        self.__plugin.cancel()

    def wait(self):
        while (
            self.__plugin.run_status.is_running()
            or self.__plugin.run_status.not_started()
        ):
            time.sleep(1)
