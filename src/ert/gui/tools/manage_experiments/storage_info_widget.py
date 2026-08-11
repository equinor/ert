from __future__ import annotations

import json
from enum import IntEnum
from typing import TYPE_CHECKING

import yaml
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QStackedLayout,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ert.gui.experiments.view import WorkflowLogView
from ert.storage import Ensemble, Experiment, RealizationStorageState

from .ensemble_widget import EnsembleWidget
from .rft_qc_widget import RftQcWidget

if TYPE_CHECKING:
    from ert.config import ErtConfig


class _WidgetType(IntEnum):
    EMPTY_WIDGET = 0
    EXPERIMENT_WIDGET = 1
    ENSEMBLE_WIDGET = 2
    REALIZATION_WIDGET = 3


class _ExperimentWidgetTabs(IntEnum):
    EXPERIMENT_TAB = 0
    OBSERVATIONS_TAB = 1
    PARAMETERS_TAB = 2
    RESPONSES_TAB = 3
    WORKFLOWS_TAB = 4


class _RealizationWidgetTabs(IntEnum):
    REALIZATION_TAB = 0
    INSPECT_RFT_TAB = 1


class _ExperimentWidget(QWidget):
    def __init__(self) -> None:
        QWidget.__init__(self)
        self._experiment: Experiment | None = None

        self._responses_text_edit = QTextEdit()
        self._responses_text_edit.setReadOnly(True)

        self._parameters_text_edit = QTextEdit()
        self._parameters_text_edit.setReadOnly(True)

        self._observations_text_edit = QTextEdit()
        self._observations_text_edit.setReadOnly(True)

        self._workflow_log_view = WorkflowLogView()

        info_frame = QFrame()
        self._name_label = QLabel()
        self._uuid_label = QLabel()
        self._created_at_label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self._name_label)
        layout.addWidget(self._uuid_label)
        layout.addWidget(self._created_at_label)
        layout.addStretch()

        info_frame.setLayout(layout)

        tab_widget = QTabWidget()
        tab_widget.insertTab(
            _ExperimentWidgetTabs.EXPERIMENT_TAB, info_frame, "Experiment"
        )
        tab_widget.insertTab(
            _ExperimentWidgetTabs.OBSERVATIONS_TAB,
            self._observations_text_edit,
            "Observations",
        )
        tab_widget.insertTab(
            _ExperimentWidgetTabs.PARAMETERS_TAB,
            self._parameters_text_edit,
            "Parameters",
        )
        tab_widget.insertTab(
            _ExperimentWidgetTabs.RESPONSES_TAB, self._responses_text_edit, "Responses"
        )
        tab_widget.insertTab(
            _ExperimentWidgetTabs.WORKFLOWS_TAB, self._workflow_log_view, "Workflows"
        )
        tab_widget.currentChanged.connect(self._current_tab_changed)
        self._tab_widget = tab_widget

        layout = QVBoxLayout()
        layout.addWidget(tab_widget)

        self.setLayout(layout)

    def _current_tab_changed(self, index: int) -> None:
        if (
            index == _ExperimentWidgetTabs.WORKFLOWS_TAB
            and self._experiment is not None
        ):
            self._workflow_log_view.load_events(self._experiment.workflow_events_path)

    @Slot(Experiment)
    def setExperiment(self, experiment: Experiment) -> None:
        self._experiment = experiment

        self._name_label.setText(f"Name: {experiment.name!s}")
        self._uuid_label.setText(f"UUID: {experiment.id!s}")

        # Determine creation time from the earliest ensemble start time
        created_text = "Unknown"
        ensemble_start_times = [ens.started_at for ens in experiment.ensembles]
        if ensemble_start_times:
            created_text = min(ensemble_start_times).strftime("%Y-%m-%d %H:%M:%S")

        self._created_at_label.setText(f"Created at: {created_text}")

        self._responses_text_edit.setText(yaml.dump(experiment.response_info, indent=4))
        self._parameters_text_edit.setText(
            json.dumps(experiment.parameter_info, indent=4)
        )
        html = "<table>"
        for obs_name in experiment.observations:
            html += f"<tr><td>{obs_name}</td></tr>"
        html += "</table>"
        self._observations_text_edit.setHtml(html)

        # The workflow tab loads lazily, so it needs a nudge when the user
        # switches experiment while already looking at it.
        self._current_tab_changed(self._tab_widget.currentIndex())


class _RealizationWidget(QWidget):
    def __init__(self, ert_config: ErtConfig | None = None) -> None:
        QWidget.__init__(self)

        info_frame = QFrame()
        self._state_label = QLabel()

        parameter_label = QLabel("Parameter states:")
        self._parameter_text_edit = QTextEdit()
        self._parameter_text_edit.setReadOnly(True)

        response_label = QLabel("Response states:")
        self._response_text_edit = QTextEdit()
        self._response_text_edit.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self._state_label)
        layout.addStretch(20)
        layout.addWidget(parameter_label)
        layout.addWidget(self._parameter_text_edit)
        layout.addStretch(20)
        layout.addWidget(response_label)
        layout.addWidget(self._response_text_edit)

        layout.addStretch()

        info_frame.setLayout(layout)

        self._tab_widget = QTabWidget()
        self._tab_widget.insertTab(
            _RealizationWidgetTabs.REALIZATION_TAB, info_frame, "Realization"
        )

        self._rft_qc_widget = RftQcWidget(ert_config=ert_config)
        self._tab_widget.insertTab(
            _RealizationWidgetTabs.INSPECT_RFT_TAB, self._rft_qc_widget, "Inspect RFT"
        )
        self._tab_widget.setTabEnabled(_RealizationWidgetTabs.INSPECT_RFT_TAB, False)
        self._rft_tab_selected = False
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        layout = QVBoxLayout()
        layout.addWidget(self._tab_widget)

        self.setLayout(layout)

    def _on_tab_changed(self, index: int) -> None:
        self._rft_tab_selected = index == _RealizationWidgetTabs.INSPECT_RFT_TAB
        if self._rft_tab_selected:
            self._rft_qc_widget.load_current_realization()

    @Slot(RealizationStorageState)
    def setRealization(self, ensemble: Ensemble, realization: int) -> None:
        has_rft = "rft" in ensemble.experiment.response_configuration
        self._tab_widget.setTabEnabled(_RealizationWidgetTabs.INSPECT_RFT_TAB, has_rft)
        if has_rft:
            self._rft_qc_widget.update_realization(
                ensemble, realization, load=self._rft_tab_selected
            )

        realization_state = ensemble.get_ensemble_state()[realization]
        self._state_label.setText(
            "Realization state: "
            f"{', '.join(sorted([s.name for s in realization_state]))}"
        )

        html = "<table>"
        for name, response_state in ensemble.get_response_state(realization).items():
            html += f"<tr><td>{name} - {response_state.name}</td></tr>"
        html += "</table>"
        self._response_text_edit.setHtml(html)

        html = "<table>"
        for name, param_state in ensemble.get_parameter_state(realization).items():
            html += f"<tr><td>{name} - {param_state.name}</td></tr>"
        html += "</table>"
        self._parameter_text_edit.setHtml(html)


class StorageInfoWidget(QWidget):
    def __init__(self, ert_config: ErtConfig | None = None) -> None:
        QWidget.__init__(self)

        self._experiment_widget = _ExperimentWidget()
        self._ensemble_widget = EnsembleWidget()
        self._realization_widget = _RealizationWidget(ert_config=ert_config)
        empty_widget = QWidget()
        self.setMinimumWidth(200)

        self._content_layout = QStackedLayout()
        self._content_layout.insertWidget(_WidgetType.EMPTY_WIDGET, empty_widget)
        self._content_layout.insertWidget(
            _WidgetType.EXPERIMENT_WIDGET, self._experiment_widget
        )
        self._content_layout.insertWidget(
            _WidgetType.ENSEMBLE_WIDGET, self._ensemble_widget
        )
        self._content_layout.insertWidget(
            _WidgetType.REALIZATION_WIDGET, self._realization_widget
        )

        layout = QVBoxLayout()
        layout.addLayout(self._content_layout)

        self.setLayout(layout)

    @Slot(Ensemble)
    def setEnsemble(self, ensemble: Ensemble) -> None:
        self._content_layout.setCurrentIndex(_WidgetType.ENSEMBLE_WIDGET)
        self._ensemble_widget.setEnsemble(ensemble)

    @Slot(Experiment)
    def setExperiment(self, experiment: Experiment) -> None:
        self._content_layout.setCurrentIndex(_WidgetType.EXPERIMENT_WIDGET)
        self._experiment_widget.setExperiment(experiment)

    @Slot(Ensemble, int)
    def setRealization(self, ensemble: Ensemble, realization: int) -> None:
        self._content_layout.setCurrentIndex(_WidgetType.REALIZATION_WIDGET)
        self._realization_widget.setRealization(ensemble, realization)
