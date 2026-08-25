from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QFormLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ert.config import ErtConfig, WarningInfo
from ert.gui.ertnotifier import ErtNotifier
from ert.gui.ertwidgets import (
    ActiveRealizationsModel,
    EnsembleSelector,
    QApplication,
    StringBox,
    Suggestor,
    TextBox,
    TextModel,
)
from ert.run_models.run_model import captured_logs
from ert.storage.local_ensemble import load_parameters_and_responses_from_runpath
from ert.validation import RangeStringArgument, StringDefinition


class LoadResultsPanel(QWidget):
    panelConfigurationChanged = Signal()

    def __init__(self, config: ErtConfig, notifier: ErtNotifier) -> None:
        QWidget.__init__(self)

        self.setMinimumWidth(600)

        self._notifier = notifier

        self._resolved_runpath = str(
            Path(config.runpath_config.runpath_format_string).resolve()
        )

        self.setWindowTitle("Load results manually")
        self.activateWindow()

        expanding_form = QFormLayout()
        runpath_label_text = "Enter runpath to load results from: "
        runpath_label = QLabel(runpath_label_text)

        self._ensemble_selector = EnsembleSelector(self._notifier)
        self._ensemble_selector.ensemble_selected.connect(self.refresh)

        self._runpath_textbox = TextBox(TextModel(self._read_current_runpath()))
        self._runpath_textbox.setPreferredHeightInLines(3)
        self._runpath_textbox.setMinimumHeight(10)
        self._runpath_textbox.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._runpath_textbox.setValidator(StringDefinition(required=["<IENS>"]))
        self._runpath_textbox.setObjectName("runpath_edit_lrm")
        self._runpath_textbox.getValidationSupport().validationChanged.connect(
            self.panelConfigurationChanged
        )
        self._runpath_textbox.textChanged.connect(self._text_change)

        expanding_form.addRow(runpath_label, self._runpath_textbox)

        fixed_form = QFormLayout()
        label_width = runpath_label.sizeHint().width()

        def make_qlabel(text: str) -> QLabel:
            """Propagate width of upper left label to other labels"""
            label = QLabel(text)
            label.setMinimumWidth(label_width)
            return label

        self.help_iens_lbl = QLabel("<IENS> will be replaced by %")

        fixed_form.addRow(make_qlabel(""), self.help_iens_lbl)
        fixed_form.addRow(make_qlabel("Load into ensemble:"), self._ensemble_selector)

        ensemble_size = config.runpath_config.num_realizations
        self._active_realizations_model = ActiveRealizationsModel(ensemble_size)
        self._active_realizations_field = StringBox(
            self._active_realizations_model,  # type: ignore
            "load_results_manually/Realizations",
        )
        self._active_realizations_field.textChanged.connect(self._text_change)
        self._active_realizations_field.setValidator(RangeStringArgument(ensemble_size))
        self._active_realizations_field.setObjectName("active_realizations_lrm")
        self.help_iens_lbl.setText(
            f"<IENS> will be replaced by {self._active_realizations_field.get_text}"
        )
        fixed_form.addRow(
            make_qlabel("Realizations to load:"), self._active_realizations_field
        )

        self._active_realizations_field.getValidationSupport().validationChanged.connect(
            self.panelConfigurationChanged
        )

        layout = QVBoxLayout()
        layout.addLayout(expanding_form, 1)
        layout.addLayout(fixed_form, 0)
        self.setLayout(layout)

    def _text_change(self) -> None:
        active_realizations = self._active_realizations_field.get_text
        self.help_iens_lbl.setText(f"<IENS> will be replaced by {active_realizations}")

    def _read_current_runpath(self) -> str:
        runpath = self._resolved_runpath
        if self._ensemble_selector.selected_ensemble:
            current_ensemble = self._ensemble_selector.selected_ensemble.name
            runpath = runpath.replace("<ERTCASE>", current_ensemble)
            runpath = runpath.replace("<ERT-CASE>", current_ensemble)
        return runpath.replace("<ITER>", "0")

    def is_configuration_valid(self) -> bool:
        return (
            self._active_realizations_field.isValid()
            and self._runpath_textbox.isValid()
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
            if self._ensemble_selector.selected_ensemble:
                write_ensemble = write_storage.get_ensemble(
                    self._ensemble_selector.selected_ensemble.id
                )
                loaded = load_parameters_and_responses_from_runpath(
                    run_path_format=self._runpath_textbox.get_text,
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
        self._runpath_textbox.setText(self._read_current_runpath())
        self._runpath_textbox.refresh()
