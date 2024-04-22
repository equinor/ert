import json
from enum import IntEnum

import numpy as np
import seaborn as sns
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedLayout,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ert.storage import Ensemble, Experiment
from ert.storage.realization_storage_state import RealizationStorageState


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


class _EnsembleWidgetTabs(IntEnum):
    ENSEMBLE_TAB = 0
    STATE_TAB = 1
    OBSERVATIONS_TAB = 2


class _ExperimentWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self._experiment = None

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

        layout = QVBoxLayout()
        layout.addWidget(tab_widget)

        self.setLayout(layout)

    @Slot(Experiment)
    def setExperiment(self, experiment: Experiment) -> None:
        self._experiment = experiment

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
        self._ensemble = None

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

        observations_frame = QFrame()

        self._observations_tree_widget = QTreeWidget(self)
        self._observations_tree_widget.currentItemChanged.connect(
            self._currentItemChanged
        )
        self._observations_tree_widget.setColumnCount(1)
        self._observations_tree_widget.setHeaderHidden(True)

        self._figure = Figure()
        self._figure.set_layout_engine("tight")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setParent(self)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.setFocus()

        layout = QHBoxLayout()
        layout.addWidget(self._observations_tree_widget)
        layout.addWidget(self._canvas)
        observations_frame.setLayout(layout)

        self._tab_widget = QTabWidget()
        self._tab_widget.insertTab(
            _EnsembleWidgetTabs.ENSEMBLE_TAB, info_frame, "Ensemble"
        )
        self._tab_widget.insertTab(
            _EnsembleWidgetTabs.STATE_TAB, self._state_text_edit, "State"
        )
        self._tab_widget.insertTab(
            _EnsembleWidgetTabs.OBSERVATIONS_TAB, observations_frame, "Observations"
        )
        self._tab_widget.currentChanged.connect(self._currentTabChanged)

        layout = QVBoxLayout()
        layout.addWidget(self._tab_widget)

        self.setLayout(layout)

    def _currentItemChanged(
        self, selected: QTreeWidgetItem, _: QTreeWidgetItem
    ) -> None:
        if not selected:
            return

        observation_name = selected.data(1, Qt.ItemDataRole.DisplayRole)
        if not observation_name:
            return

        observation_label = selected.data(0, Qt.ItemDataRole.DisplayRole)
        observations_dict = self._ensemble.experiment.observations

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_title(observation_name)
        ax.grid(True)

        observation_ds = observations_dict[observation_name]

        response_name = observation_ds.attrs["response"]
        response_ds = self._ensemble.load_responses(
            response_name,
            tuple(self._ensemble.get_realization_list_with_responses()),
        )

        # check if the response is empty
        if bool(response_ds.dims):
            if response_name == "summary":
                response_ds = response_ds.sel(name=str(observation_ds.name.data[0]))

            if "time" in observation_ds.coords:
                observation_ds = observation_ds.sel(time=observation_label)
                response_ds = response_ds.sel(time=observation_label)
            elif "index" in observation_ds.coords:
                observation_ds = observation_ds.sel(index=int(observation_label))
                response_ds = response_ds.drop(["index"]).sel(
                    index=int(observation_label)
                )

            ax.errorbar(
                x="Observation",
                y=observation_ds.get("observations"),
                yerr=observation_ds.get("std"),
                fmt=".",
                linewidth=1,
                capsize=4,
                color="black",
            )

            response_ds = response_ds.rename_vars({"values": "Responses"})
            sns.boxplot(response_ds.to_dataframe(), ax=ax)
            sns.stripplot(response_ds.to_dataframe(), ax=ax, size=4, color=".3")

        else:
            if "time" in observation_ds.coords:
                observation_ds = observation_ds.sel(time=observation_label)
            elif "index" in observation_ds.coords:
                observation_ds = observation_ds.sel(index=int(observation_label))

            ax.errorbar(
                x="Observation",
                y=observation_ds.get("observations"),
                yerr=observation_ds.get("std"),
                fmt=".",
                linewidth=1,
                capsize=4,
                color="black",
            )

        self._canvas.draw()

    def _currentTabChanged(self, index: int) -> None:
        if index == _EnsembleWidgetTabs.STATE_TAB:
            self._state_text_edit.clear()
            html = "<table>"
            for state_index, value in enumerate(self._ensemble.get_ensemble_state()):
                html += (
                    f"<tr><td width=30>{state_index:d}.</td><td>{value.name}</td></tr>"
                )
            html += "</table>"
            self._state_text_edit.setHtml(html)

        elif index == _EnsembleWidgetTabs.OBSERVATIONS_TAB:
            self._observations_tree_widget.clear()
            self._figure.clear()
            self._canvas.draw()

            observations_dict = self._ensemble.experiment.observations
            for obs_name, obs_ds in observations_dict.items():
                response_name = obs_ds.attrs["response"]
                if response_name == "summary":
                    name = obs_ds.name.data[0]
                else:
                    name = response_name

                match_list = self._observations_tree_widget.findItems(
                    name, Qt.MatchFlag.MatchExactly
                )
                if len(match_list) == 0:
                    root = QTreeWidgetItem(self._observations_tree_widget, [name])
                else:
                    root = match_list[0]

                if "time" in obs_ds.coords:
                    for t in obs_ds.time:
                        QTreeWidgetItem(
                            root,
                            [str(np.datetime_as_string(t.values, unit="D")), obs_name],
                        )
                elif "index" in obs_ds.coords:
                    for t in obs_ds.index:
                        QTreeWidgetItem(root, [str(t.data), obs_name])

                self._observations_tree_widget.sortItems(0, Qt.SortOrder.AscendingOrder)

    @Slot(Ensemble)
    def setEnsemble(self, ensemble: Ensemble) -> None:
        self._ensemble = ensemble

        self._name_label.setText(f"Name: {str(ensemble.name)}")
        self._uuid_label.setText(f"UUID: {str(ensemble.id)}")

        self._tab_widget.setCurrentIndex(0)


class _RealizationWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        info_frame = QFrame()
        self._state_label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self._state_label)
        layout.addStretch()

        info_frame.setLayout(layout)

        tab_widget = QTabWidget()
        tab_widget.addTab(info_frame, "Realization")

        layout = QVBoxLayout()
        layout.addWidget(tab_widget)

        self.setLayout(layout)

    @Slot(RealizationStorageState)
    def setRealization(self, realization_state: RealizationStorageState) -> None:
        self._state_label.setText(f"Realization state: {realization_state.name}")


class StorageInfoWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self._experiment_widget = _ExperimentWidget()
        self._ensemble_widget = _EnsembleWidget()
        self._realization_widget = _RealizationWidget()
        empty_widget = QWidget()

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

    @Slot(RealizationStorageState)
    def setRealization(self, realization_state: RealizationStorageState):
        self._content_layout.setCurrentIndex(_WidgetType.REALIZATION_WIDGET)
        self._realization_widget.setRealization(realization_state)
