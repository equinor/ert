import uuid
from contextlib import contextmanager
from typing import Optional

import numpy as np
from qtpy.QtCore import QObject, Qt, QThread, Signal, Slot
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication, QMessageBox

from ert.analysis import ErtAnalysisError, smoother_update
from ert.analysis._es_update import UpdateSettings
from ert.enkf_main import EnKFMain, _seed_sequence
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets.statusdialog import StatusDialog
from ert.gui.tools import Tool
from ert.gui.tools.run_analysis import RunAnalysisPanel
from ert.storage import EnsembleAccessor, EnsembleReader, StorageAccessor


class Analyse(QObject):
    finished = Signal(str, str)
    """Signal(Optional[str], str)
    Note: first argument is optional even though it is not declared that way

    Signals emitted contain:
    first arg -- optional error string
    second arg -- always returns source_fs.name"""
    progress_update = Signal(object)
    """Signal(Progress)"""

    def __init__(
        self,
        ert: EnKFMain,
        target_fs: EnsembleAccessor,
        source_fs: EnsembleReader,
    ):
        QObject.__init__(self)
        self._ert = ert
        self._target_fs = target_fs
        self._source_fs = source_fs

    @Slot()
    def run(self):
        """Runs analysis using target and source cases. Returns whether
        the analysis was successful."""
        error: Optional[str] = None
        config = self._ert.ert_config
        rng = np.random.default_rng(_seed_sequence(config.random_seed))
        update_settings = UpdateSettings(
            std_cutoff=config.analysis_config.std_cutoff,
            alpha=config.analysis_config.enkf_alpha,
            misfit_preprocess=False,
            min_required_realizations=config.analysis_config.minimum_required_realizations,
        )
        try:
            smoother_update(
                self._source_fs,
                self._target_fs,
                str(uuid.uuid4()),
                self._ert.update_configuration,
                update_settings,
                config.analysis_config.es_module,
                rng,
                self.progress_callback,
                log_path=config.analysis_config.log_path,
            )
        except ErtAnalysisError as e:
            error = str(e)
        except Exception as e:
            error = f"Unknown exception occurred with error: {str(e)}"

        self.finished.emit(error, self._source_fs.name)

    def progress_callback(self, progress: object):
        self.progress_update.emit(progress)


class RunAnalysisTool(Tool):
    def __init__(self, ert: EnKFMain, notifier: ErtNotifier):
        super().__init__("Run analysis", QIcon("img:formula.svg"))
        self.ert = ert
        self.notifier = notifier
        self._run_widget: Optional[RunAnalysisPanel] = None
        self._dialog: Optional[StatusDialog] = None
        self._thread: Optional[QThread] = None
        self._analyse: Optional[Analyse] = None

    def trigger(self):
        if self._run_widget is None:
            self._run_widget = RunAnalysisPanel(
                self.ert.ert_config.analysis_config.es_module,
                self.ert.ert_config.model_config.num_realizations,
                self.notifier,
            )
        if self._dialog is None:
            self._dialog = StatusDialog(
                "Run analysis",
                self._run_widget,
                self.parent(),
            )
            self._dialog.run.connect(self.run)
            self._dialog.exec_()
        else:
            self._run_widget.target_case_text.clear()
            self._dialog.show()

    def run(self):
        target: str = self._run_widget.target_case()
        if len(target.strip()) == 0:
            self._report_empty_target()
            return

        self._enable_dialog(False)
        try:
            self._init_analyse(self._run_widget.source_case(), target)
            self._init_and_start_thread()
        except Exception as e:
            self._enable_dialog(True)
            QMessageBox.critical(None, "Error", str(e))

    def _on_finished(self, error: Optional[str], case_name: str):
        self._enable_dialog(True)

        if not error:
            QMessageBox.information(
                None,
                "Analysis finished",
                f"Successfully ran analysis for case '{case_name}'.",
            )
        else:
            QMessageBox.warning(
                None,
                "Failed",
                f"Unable to run analysis for case '{case_name}'.\n"
                f"The following error occurred: {error}",
            )
            return

        self.notifier.ertChanged.emit()
        self._dialog.accept()

    def _init_and_start_thread(self):
        self._thread = QThread()

        self._analyse.moveToThread(self._thread)
        self._thread.started.connect(self._analyse.run)
        self._analyse.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._analyse.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._analyse.finished.connect(self._on_finished)
        self._analyse.finished.connect(self._dialog.clear_status)
        self._analyse.progress_update.connect(self._dialog.progress_update)

        self._thread.start()

    @staticmethod
    def _report_empty_target():
        QMessageBox.warning(None, "Invalid target", "Target case can not be empty")

    def _enable_dialog(self, enable: bool):
        self._dialog.enable_buttons(enable)
        if enable:
            QApplication.restoreOverrideCursor()
        else:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _init_analyse(self, source_fs: EnsembleReader, target: str):
        with add_context_to_error(
            "Could not get StorageAccessor for writing to ensemble"
        ):
            accessor: StorageAccessor = self.notifier.storage.to_accessor()
        target_fs: EnsembleAccessor = accessor.create_ensemble(
            source_fs.experiment_id,
            name=target,
            ensemble_size=source_fs.ensemble_size,
            iteration=source_fs.iteration + 1,
            prior_ensemble=source_fs,
        )

        self._analyse = Analyse(
            self.ert,
            target_fs,
            source_fs,
        )


@contextmanager
def add_context_to_error(msg: str):
    try:
        yield
    except Exception as e:
        text = f"{msg}:\n{e.args[0]}" if e.args else str(msg)
        e.args = (text,) + e.args[1:]
        raise
