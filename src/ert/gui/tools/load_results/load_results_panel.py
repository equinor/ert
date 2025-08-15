from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import QFormLayout, QLabel, QMessageBox, QWidget

from ert.config import ErtConfig, WarningInfo
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    ActiveRealizationsModel,
    EnsembleSelector,
    QApplication,
    StringBox,
    TextBox,
    TextModel,
)
from ert.gui.suggestor import Suggestor
from ert.run_models.run_model import captured_logs
from ert.storage.local_ensemble import load_parameters_and_responses_from_runpath
from ert.validation import RangeStringArgument, StringDefinition


class LoadResultsPanel(QWidget):
    panelConfigurationChanged = Signal()

    def __init__(self, config: ErtConfig, notifier: ErtNotifier) -> None:
        QWidget.__init__(self)

        self.setMinimumWidth(500)
        self.setMinimumHeight(200)
        self._dynamic = False
        self._notifier = notifier

        self._resolved_run_path = str(
            Path(config.runpath_config.runpath_format_string).resolve()
        )

        self.setWindowTitle("Load results manually")
        self.activateWindow()

        layout = QFormLayout()

        self._run_path_text = TextBox(TextModel(self.readCurrentRunPath()))
        self._run_path_text.setFixedHeight(80)
        self._run_path_text.setValidator(StringDefinition(required=["<IENS>"]))
        self._run_path_text.setObjectName("run_path_edit_lrm")
        self._run_path_text.getValidationSupport().validationChanged.connect(
            self.panelConfigurationChanged
        )
        self._run_path_text.textChanged.connect(self.text_change)

        self.help_iter_lbl = QLabel("<ITER> will be replace by: 0")
        self.help_iens_lbl = QLabel("<IENS> will be replace by %")
        layout.addRow("Load data from run path: ", self._run_path_text)
        ensemble_selector = EnsembleSelector(self._notifier)
        layout.addRow("", self.help_iens_lbl)
        layout.addRow("", self.help_iter_lbl)
        layout.addRow("Load into ensemble:", ensemble_selector)
        self._ensemble_selector = ensemble_selector

        ensemble_size = config.runpath_config.num_realizations
        self._active_realizations_model = ActiveRealizationsModel(ensemble_size)
        self._active_realizations_field = StringBox(
            self._active_realizations_model,  # type: ignore
            "load_results_manually/Realizations",
        )
        self._active_realizations_field.textChanged.connect(self.text_change)
        self._active_realizations_field.setValidator(RangeStringArgument(ensemble_size))
        self._active_realizations_field.setObjectName("active_realizations_lrm")
        self.help_iens_lbl.setText(
            f"<IENS> will be replace by {self._active_realizations_field.get_text}"
        )
        layout.addRow("Realizations to load:", self._active_realizations_field)

        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.panelConfigurationChanged
        )
        self.setLayout(layout)

    def text_change(self) -> None:
        active_realizations = self._active_realizations_field.get_text
        self.help_iens_lbl.setText(f"<IENS> will be replace by {active_realizations}")
        self.help_iter_lbl.setVisible("<ITER>" in self._run_path_text.get_text)

    def readCurrentRunPath(self) -> str:
        current_ensemble = self._notifier.current_ensemble_name
        run_path = self._resolved_run_path
        run_path = run_path.replace("<ERTCASE>", current_ensemble)
        run_path = run_path.replace("<ERT-CASE>", current_ensemble)
        return run_path

    def isConfigurationValid(self) -> bool:
        return (
            self._active_realizations_field.isValid() and self._run_path_text.isValid()
        )

    def load(self) -> int:
        realizations = self._active_realizations_model.getActiveRealizationsMask()
        active_realizations = [
            iens for iens, active in enumerate(realizations) if active
        ]
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        messages: list[str] = []
        loaded: int = 0
        with captured_logs(messages), self._notifier.write_storage() as write_storage:
            if self._notifier.current_ensemble is not None:
                write_ensemble = write_storage.get_ensemble(
                    self._notifier.current_ensemble.id
                )
                loaded = load_parameters_and_responses_from_runpath(
                    run_path_format=self._run_path_text.get_text,
                    ensemble=write_ensemble,
                    active_realizations=active_realizations,
                )
        QApplication.restoreOverrideCursor()

        if loaded == realizations.count(True):
            QMessageBox.information(
                self, "Success", "Successfully loaded all realizations"
            )
        else:
            txt = "No realizations loaded\n" + "\n".join(messages)

            if loaded > 0:
                txt = f"Successfully loaded {loaded} realization(s)\n" + "\n".join(
                    messages
                )

            fail_msg_box = Suggestor(
                errors=[],
                warnings=[WarningInfo(message=txt)],
                deprecations=[],
                continue_action=None,
                widget_info="""\
                               <p style="font-size: 28px;">ERT experiment failed!</p>
                               <p style="font-size: 16px;">These errors were detected:
                               </p>
                           """,
                parent=self,
            )
            fail_msg_box.show()

        return loaded

    def refresh(self) -> None:
        self._run_path_text.setText(self.readCurrentRunPath())
        self._run_path_text.refresh()
