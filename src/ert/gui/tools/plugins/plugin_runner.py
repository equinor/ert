from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from _ert.threading import ErtThread
from ert.config import CancelPluginException
from ert.workflow_runner import WorkflowJobRunner

from .process_job_dialog import ProcessJobDialog

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.storage import LocalStorage

    from .plugin import Plugin


class PluginRunner:
    def __init__(
        self, plugin: "Plugin", ert_config: ErtConfig, storage: LocalStorage
    ) -> None:
        super().__init__()
        self.ert_config = ert_config
        self.storage = storage
        self.__plugin = plugin

        self.__plugin_finished_callback: Callable[[], None] = lambda: None

        self.__result = None
        self._runner = WorkflowJobRunner(plugin.getWorkflowJob())
        self.poll_thread: Optional[ErtThread] = None

    def run(self) -> None:
        try:
            plugin = self.__plugin

            arguments = plugin.getArguments(
                fixtures={"storage": self.storage, "ert_config": self.ert_config}
            )
            dialog = ProcessJobDialog(plugin.getName(), plugin.getParentWindow())
            dialog.setObjectName("process_job_dialog")

            dialog.cancelConfirmed.connect(self.cancel)
            fixtures = {
                k: getattr(self, k)
                for k in ["storage", "ert_config"]
                if getattr(self, k)
            }
            workflow_job_thread = ErtThread(
                name="ert_gui_workflow_job_thread",
                target=self.__runWorkflowJob,
                args=(arguments, fixtures),
                daemon=True,
                should_raise=False,
            )
            workflow_job_thread.start()

            self.poll_thread = ErtThread(
                name="ert_gui_workflow_job_poll_thread",
                target=self.__pollRunner,
                args=(dialog,),
                daemon=True,
                should_raise=False,
            )
            self.poll_thread.start()

            dialog.show()
        except CancelPluginException:
            print("Plugin cancelled before execution!")

    def __runWorkflowJob(
        self, arguments: Optional[List[Any]], fixtures: Dict[str, Any]
    ) -> None:
        self.__result = self._runner.run(arguments, fixtures=fixtures)

    def __pollRunner(self, dialog: ProcessJobDialog) -> None:
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

    def cancel(self) -> None:
        if self.isRunning():
            self._runner.cancel()

    def wait(self) -> None:
        while self.isRunning():
            time.sleep(1)

    def setPluginFinishedCallback(self, callback: Callable[[], None]) -> None:
        self.__plugin_finished_callback = callback
