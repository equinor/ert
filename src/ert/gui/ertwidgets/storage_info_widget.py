import json
from enum import IntEnum

import yaml
from qtpy.QtCore import Slot
from qtpy.QtWidgets import (
    QFrame,
    QLabel,
    QStackedLayout,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ert.storage import Ensemble, Experiment


class _WidgetType(IntEnum):
    EMPTY_WIDGET = 0
    EXPERIMENT_WIDGET = 1
    ENSEMBLE_WIDGET = 2


class _ExperimentWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self._responses_text_edit = QTextEdit()
        self._responses_text_edit.setReadOnly(True)

        self._parameters_text_edit = QTextEdit()
        self._parameters_text_edit.setReadOnly(True)

        self._observations_text_edit = QTextEdit()
        self._observations_text_edit.setReadOnly(True)

        info_frame = QFrame()
        self._name_label = QLabel()
        self._uuid_label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self._name_label)
        layout.addWidget(self._uuid_label)
        layout.addStretch()

        info_frame.setLayout(layout)

        tab_widget = QTabWidget()
        tab_widget.addTab(info_frame, "Experiment")
        tab_widget.addTab(self._observations_text_edit, "Observations")
        tab_widget.addTab(self._parameters_text_edit, "Parameters")
        tab_widget.addTab(self._responses_text_edit, "Responses")

        layout = QVBoxLayout()
        layout.addWidget(tab_widget)

        self.setLayout(layout)

    @Slot(Experiment)
    def setExperiment(self, experiment: Experiment) -> None:
        self._name_label.setText(f"Name: {str(experiment.name)}")
        self._uuid_label.setText(f"UUID: {str(experiment.id)}")

        self._responses_text_edit.setText(yaml.dump(experiment.response_info, indent=4))
        self._parameters_text_edit.setText(
            json.dumps(experiment.parameter_info, indent=4)
        )
        html = "<table>"
        for obs_name in experiment.observations:
            html += f"<tr><td>{obs_name}</td></tr>"
        html += "</table>"
        self._observations_text_edit.setHtml(html)


class _EnsembleWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        info_frame = QFrame()
        self._name_label = QLabel()
        self._uuid_label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self._name_label)
        layout.addWidget(self._uuid_label)
        layout.addStretch()

        info_frame.setLayout(layout)

        self._state_text_edit = QTextEdit()
        self._state_text_edit.setReadOnly(True)
        self._state_text_edit.setObjectName("ensemble_state_text")

        tab_widget = QTabWidget()
        tab_widget.addTab(info_frame, "Ensemble")
        tab_widget.addTab(self._state_text_edit, "State")

        layout = QVBoxLayout()
        layout.addWidget(tab_widget)

        self.setLayout(layout)

    @Slot(Ensemble)
    def setEnsemble(self, ensemble: Ensemble) -> None:
        self._name_label.setText(f"Name: {str(ensemble.name)}")
        self._uuid_label.setText(f"UUID: {str(ensemble.id)}")

        self._state_text_edit.clear()
        html = "<table>"
        for state_index, value in enumerate(ensemble.get_ensemble_state()):
            html += f"<tr><td width=30>{state_index:d}.</td><td>{value.name}</td></tr>"
        html += "</table>"
        self._state_text_edit.setHtml(html)


class StorageInfoWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self._experiment_widget = _ExperimentWidget()
        self._ensemble_widget = _EnsembleWidget()
        empty_widget = QWidget()

        self._content_layout = QStackedLayout()
        self._content_layout.insertWidget(_WidgetType.EMPTY_WIDGET, empty_widget)
        self._content_layout.insertWidget(
            _WidgetType.EXPERIMENT_WIDGET, self._experiment_widget
        )
        self._content_layout.insertWidget(
            _WidgetType.ENSEMBLE_WIDGET, self._ensemble_widget
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
