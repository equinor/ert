from __future__ import annotations

import functools
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np
from qtpy.QtCore import QObject, Qt, QThread, Signal, Slot
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication, QMessageBox

from ert.analysis import ErtAnalysisError, smoother_update
from ert.analysis.event import AnalysisEvent, AnalysisStatusEvent, AnalysisTimeEvent
from ert.enkf_main import _seed_sequence
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.statusdialog import StatusDialog
from ert.gui.tools import Tool
from ert.gui.tools.run_analysis import RunAnalysisPanel
from ert.run_models.event import RunModelEvent, RunModelStatusEvent, RunModelTimeEvent
from ert.storage import Ensemble

if TYPE_CHECKING:
    from ert.config import ErtConfig


class Analyse(QObject):
    finished = Signal(str, str)
    """Signal(Optional[str], str)
    Note: first argument is optional even though it is not declared that way

    Signals emitted contain:
    first arg -- optional error string
    second arg -- always returns source_ensemble.name"""
    progress_update = Signal(RunModelEvent)
    """Signal(Progress)"""

    def __init__(
        self,
        ert_config: ErtConfig,
        target_ensemble: Ensemble,
        source_ensemble: Ensemble,
    ) -> None:
        QObject.__init__(self)
        self._ert_config = ert_config
        self._target_ensemble = target_ensemble
        self._source_ensemble = source_ensemble

    @Slot()
    def run(self) -> None:
        """Runs analysis using target and source ensembles. Returns whether
        the analysis was successful."""
        error: Optional[str] = None
        config = self._ert_config
        rng = np.random.default_rng(_seed_sequence(config.random_seed))
        update_settings = config.analysis_config.observation_settings
        update_id = uuid.uuid4()
        try:
            smoother_update(
                self._source_ensemble,
                self._target_ensemble,
                self._source_ensemble.experiment.observations.keys(),
                self._source_ensemble.experiment.update_parameters,
                update_settings,
                config.analysis_config.es_module,
                rng,
                functools.partial(
                    self.send_smoother_event,
                    update_id,
                ),
            )
        except ErtAnalysisError as e:
            error = str(e)
        except Exception as e:
            error = f"Unknown exception occurred with error: {str(e)}"

        self.finished.emit(error, self._source_ensemble.name)

    def send_smoother_event(self, run_id: uuid.UUID, event: AnalysisEvent) -> None:
        if isinstance(event, AnalysisStatusEvent):
            self.progress_update.emit(
                RunModelStatusEvent(iteration=0, run_id=run_id, msg=event.msg)
            )
        elif isinstance(event, AnalysisTimeEvent):
            self.progress_update.emit(
                RunModelTimeEvent(
                    iteration=0,
                    run_id=run_id,
                    elapsed_time=event.elapsed_time,
                    remaining_time=event.remaining_time,
                )
            )


class RunAnalysisTool(Tool):
    def __init__(self, config: ErtConfig, notifier: ErtNotifier) -> None:
        super().__init__("Run analysis", QIcon("img:formula.svg"))
        self.ert_config = config
        self.notifier = notifier
        self._run_widget: Optional[RunAnalysisPanel] = None
        self._dialog: Optional[StatusDialog] = None
        self._thread: Optional[QThread] = None
        self._analyse: Optional[Analyse] = None

    def trigger(self) -> None:
        if self._run_widget is None:
            self._run_widget = RunAnalysisPanel(
                self.ert_config.analysis_config.es_module,
                self.ert_config.model_config.num_realizations,
                self.notifier,
            )
        if self._dialog is None:
            self._dialog = StatusDialog(
                "Run analysis",
                self._run_widget,
                self.parent(),  # type: ignore
            )
            self._dialog.run.connect(self.run)
            self._dialog.exec_()
        else:
            self._run_widget.target_ensemble_text.clear()
            self._dialog.show()

    def run(self) -> None:
        assert self._run_widget is not None
        target: str = self._run_widget.target_ensemble()
        if len(target.strip()) == 0:
            self._report_empty_target()
            return

        self._enable_dialog(False)
        try:
            self._init_analyse(self._run_widget.source_ensemble(), target)
            self._init_and_start_thread()
        except Exception as e:
            self._enable_dialog(True)
            QMessageBox.critical(None, "Error", str(e))

    def _on_finished(self, error: Optional[str], ensemble_name: str) -> None:
        self._enable_dialog(True)

        if not error:
            QMessageBox.information(
                None,
                "Analysis finished",
                f"Successfully ran analysis for ensemble '{ensemble_name}'.",
            )
        else:
            QMessageBox.warning(
                None,
                "Failed",
                f"Unable to run analysis for ensemble '{ensemble_name}'.\n"
                f"The following error occurred: {error}",
            )
            return

        self.notifier.ertChanged.emit()
        assert self._dialog is not None
        self._dialog.accept()

    def _init_and_start_thread(self) -> None:
        self._thread = QThread()

        assert self._analyse is not None
        self._analyse.moveToThread(self._thread)
        self._thread.started.connect(self._analyse.run)
        self._analyse.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._analyse.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._analyse.finished.connect(self._on_finished)
        assert self._dialog is not None
        self._analyse.finished.connect(self._dialog.clear_status)
        self._analyse.progress_update.connect(self._dialog.progress_update)

        self._thread.start()

    @staticmethod
    def _report_empty_target() -> None:
        QMessageBox.warning(None, "Invalid target", "Target ensemble can not be empty")

    def _enable_dialog(self, enable: bool) -> None:
        assert self._dialog is not None
        self._dialog.enable_buttons(enable)
        if enable:
            QApplication.restoreOverrideCursor()
        else:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _init_analyse(self, source_ensemble: Ensemble, target: str) -> None:
        target_ensemble = self.notifier.storage.create_ensemble(
            source_ensemble.experiment_id,
            name=target,
            ensemble_size=source_ensemble.ensemble_size,
            iteration=source_ensemble.iteration + 1,
            prior_ensemble=source_ensemble,
        )

        self._analyse = Analyse(
            self.ert_config,
            target_ensemble,
            source_ensemble,
        )


@contextmanager
def add_context_to_error(msg: str) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        text = f"{msg}:\n{e.args[0]}" if e.args else str(msg)
        e.args = (text,) + e.args[1:]
        raise
