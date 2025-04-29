from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from _ert.threading import ErtThread
from ert.config import ErtConfig
from ert.plugins import CancelPluginException, WorkflowFixtures
from ert.runpaths import Runpaths
from ert.workflow_runner import WorkflowJobRunner

from .process_job_dialog import ProcessJobDialog

if TYPE_CHECKING:
    from ert.storage import LocalStorage

    from .plugin import Plugin


class PluginRunner:
    def __init__(
        self, plugin: Plugin, ert_config: ErtConfig, storage: LocalStorage
    ) -> None:
        super().__init__()

        self.ert_config = ert_config
        self.storage = storage
        self.__plugin = plugin
        self.__plugin_finished_callback: Callable[[], None] = lambda: None

        self.__result = None
        self._runner = WorkflowJobRunner(plugin.getWorkflowJob())
        self.poll_thread: ErtThread | None = None

    def run(self) -> None:
        ert_config = self.ert_config
        try:
            plugin = self.__plugin
            run_paths = Runpaths(
                jobname_format=ert_config.runpath_config.jobname_format_string,
                runpath_format=ert_config.runpath_config.runpath_format_string,
                filename=str(ert_config.runpath_file),
                substitutions=ert_config.substitutions,
                eclbase=ert_config.runpath_config.eclbase_format_string,
            )
            arguments = plugin.getArguments(
                fixtures={
                    "storage": self.storage,
                    "random_seed": ert_config.random_seed,
                    "reports_dir": str(
                        self.ert_config.analysis_config.log_path / "reports"
                    ),
                    "observation_settings": ert_config.analysis_config.observation_settings,  # noqa: E501
                    "es_settings": ert_config.analysis_config.es_settings,
                    "run_paths": run_paths,
                }
            )
            dialog = ProcessJobDialog(plugin.getName(), plugin.getParentWindow())
            dialog.setObjectName("process_job_dialog")

            dialog.cancelConfirmed.connect(self.cancel)
            fixtures = {"storage": self.storage, "run_paths": run_paths}
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
        self, arguments: list[Any] | None, fixtures: WorkflowFixtures
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
