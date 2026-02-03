import contextlib
import json
from enum import IntEnum

import polars as pl
import seaborn as sns
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvas  # type: ignore
from matplotlib.figure import Figure
from pandas.errors import PerformanceWarning
from polars import DataFrame
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ert import LibresFacade
from ert.storage import Ensemble, Experiment, RealizationStorageState
from ert.warnings import capture_specific_warning

from .export_dialog import ExportDialog


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
    PARAMETERS_TAB = 3
    MISFIT_TAB = 4


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

        self._name_label.setText(f"Name: {experiment.name!s}")
        self._uuid_label.setText(f"UUID: {experiment.id!s}")

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
    def __init__(self) -> None:
        QWidget.__init__(self)
        self._ensemble: Ensemble | None = None

        info_frame = QFrame()
        self._name_label = QLabel()
        self._uuid_label = QLabel()

        info_layout = QVBoxLayout()
        info_layout.addWidget(self._name_label)
        info_layout.addWidget(self._uuid_label)
        info_layout.addStretch()

        info_frame.setLayout(info_layout)

        state_frame = QFrame()
        state_layout = QHBoxLayout()
        self._state_text_edit = QTextEdit()
        self._state_text_edit.setReadOnly(True)
        self._state_text_edit.setObjectName("ensemble_state_text")
        state_layout.addWidget(self._state_text_edit)
        state_frame.setLayout(state_layout)

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

        observations_layout = QHBoxLayout()
        observations_layout.addWidget(self._observations_tree_widget)
        observations_layout.addWidget(self._canvas)
        observations_frame.setLayout(observations_layout)

        self._parameters_table = QTableWidget()
        self._parameters_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._export_params_button = QPushButton("Export...")
        self._export_params_button.clicked.connect(self.onClickExportParameters)

        parameters_frame = self.create_export_frame(
            self._parameters_table, self._export_params_button
        )

        self._misfit_table = QTableWidget()
        self._export_misfit_button = QPushButton("Export...")
        self._export_misfit_button.clicked.connect(self.onClickExportMisfit)

        misfit_frame = self.create_export_frame(
            self._misfit_table, self._export_misfit_button
        )

        self._tab_widget = QTabWidget()
        self._tab_widget.insertTab(
            _EnsembleWidgetTabs.ENSEMBLE_TAB, info_frame, "Ensemble"
        )
        self._tab_widget.insertTab(_EnsembleWidgetTabs.STATE_TAB, state_frame, "State")
        self._tab_widget.insertTab(
            _EnsembleWidgetTabs.OBSERVATIONS_TAB, observations_frame, "Observations"
        )
        self._tab_widget.insertTab(
            _EnsembleWidgetTabs.PARAMETERS_TAB, parameters_frame, "Parameters"
        )
        self._tab_widget.insertTab(
            _EnsembleWidgetTabs.MISFIT_TAB, misfit_frame, "Misfit"
        )
        self._tab_widget.currentChanged.connect(self._currentTabChanged)

        layout = QVBoxLayout()
        layout.addWidget(self._tab_widget)

        self.setLayout(layout)

    def create_export_frame(self, table: QTableWidget, button: QPushButton) -> QFrame:
        export_frame = QFrame()
        export_layout = QVBoxLayout()
        vertical_header = table.verticalHeader()
        assert vertical_header is not None
        vertical_header.setVisible(False)
        export_layout.addWidget(table)
        export_layout.addWidget(button)
        export_frame.setLayout(export_layout)
        return export_frame

    def _currentItemChanged(
        self, selected: QTreeWidgetItem, _: QTreeWidgetItem
    ) -> None:
        if not selected:
            return

        observation_key = selected.data(1, Qt.ItemDataRole.DisplayRole)
        parent = selected.parent()

        if not observation_key or not parent:
            return

        response_type = parent.data(0, Qt.ItemDataRole.UserRole)
        observation_label = selected.data(0, Qt.ItemDataRole.DisplayRole)
        assert self._ensemble is not None
        observations_dict = self._ensemble.experiment.observations

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_title(observation_key)
        ax.grid(True)

        obs_for_type = observations_dict[response_type]

        response_config = self._ensemble.experiment.response_configuration[
            response_type
        ]
        x_axis_col = response_config.primary_key[-1]

        def _filter_on_observation_label(df: pl.DataFrame) -> pl.DataFrame:
            # We add a column with the display name of the x axis column
            # to correctly compare it to the observation_label
            # (which is also such a display name)
            return df.with_columns(
                df[x_axis_col]
                .map_elements(
                    lambda x: response_config.display_column(x, x_axis_col),
                )
                .cast(pl.String)
                .alias("temp")
            ).filter(pl.col("temp").eq(observation_label))[
                [x for x in df.columns if x != "temp"]
            ]

        obs = obs_for_type.filter(pl.col("observation_key").eq(observation_key))
        obs = _filter_on_observation_label(obs)

        response_key = obs["response_key"].unique().to_list()[0]
        reals_with_responses = tuple(
            self._ensemble.get_realization_list_with_responses()
        )

        response_ds = (
            self._ensemble.load_responses(
                response_key,
                reals_with_responses,
            )
            if reals_with_responses
            else None
        )

        scaling_df = self._ensemble.load_observation_scaling_factors()

        def _try_render_scaled_obs() -> None:
            if scaling_df is None:
                return None

            index_col = pl.concat_str(response_config.primary_key, separator=", ")
            joined = obs.with_columns(index_col.alias("_tmp_index")).join(
                scaling_df,
                how="left",
                left_on=["observation_key", "_tmp_index"],
                right_on=["obs_key", "index"],
            )[["observations", "std", "scaling_factor"]]

            joined_small = joined[["observations", "std", "scaling_factor"]]
            joined_small = joined_small.group_by(["observations", "std"]).agg(
                [pl.col("scaling_factor").product()]
            )
            joined_small = joined_small.with_columns(
                joined_small["std"] * joined_small["scaling_factor"]
            )

            ax.errorbar(
                x="Scaled observation",
                y=joined_small["observations"].to_list(),
                yerr=joined_small["std"].to_list(),
                fmt=".",
                linewidth=1,
                capsize=4,
                color="black",
            )

        if response_ds is not None and not response_ds.is_empty():
            response_ds_for_label = _filter_on_observation_label(response_ds).rename(
                {"values": "Responses"}
            )[["response_key", "Responses"]]

            ax.errorbar(
                x="Observation",
                y=obs["observations"],
                yerr=obs["std"],
                fmt=".",
                linewidth=1,
                capsize=4,
                color="black",
            )
            _try_render_scaled_obs()

            sns.boxplot(response_ds_for_label.to_pandas(), ax=ax)
            sns.stripplot(response_ds_for_label.to_pandas(), ax=ax, size=4, color=".3")

        else:
            ax.errorbar(
                x="Observation",
                y=obs["observations"],
                yerr=obs["std"],
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
            assert self._ensemble is not None
            self._ensemble.refresh_ensemble_state()
            for state_index, state in enumerate(self._ensemble.get_ensemble_state()):
                html += (
                    f"<tr><td width=30>{state_index:d}.</td>"
                    f"<td>{', '.join([s.name for s in state])}</td></tr>"
                )
            html += "</table>"
            self._state_text_edit.setHtml(html)

        elif index == _EnsembleWidgetTabs.OBSERVATIONS_TAB:
            self._observations_tree_widget.clear()
            self._figure.clear()
            self._canvas.draw()

            assert self._ensemble is not None
            exp = self._ensemble.experiment

            for response_type, obs_ds_for_type in exp.observations.items():
                for obs_key, response_key in (
                    obs_ds_for_type.select(["observation_key", "response_key"])
                    .unique()
                    .to_numpy()
                ):
                    match_list = self._observations_tree_widget.findItems(
                        response_key, Qt.MatchFlag.MatchExactly, 0
                    )

                    root = next(
                        (
                            item
                            for item in match_list
                            if item.data(0, Qt.ItemDataRole.UserRole) == response_type
                        ),
                        None,
                    )

                    if root is None:
                        root = QTreeWidgetItem(
                            self._observations_tree_widget, [response_key]
                        )
                        root.setData(0, Qt.ItemDataRole.UserRole, response_type)

                    obs_ds = obs_ds_for_type.filter(
                        pl.col("observation_key").eq(obs_key)
                    )
                    response_config = exp.response_configuration[response_type]
                    column_to_display = response_config.primary_key[-1]
                    for t in obs_ds[column_to_display].to_list():
                        QTreeWidgetItem(
                            root,
                            [
                                response_config.display_column(t, column_to_display),
                                obs_key,
                            ],
                        )

            self._observations_tree_widget.sortItems(0, Qt.SortOrder.AscendingOrder)

            for i in range(self._observations_tree_widget.topLevelItemCount()):
                item = self._observations_tree_widget.topLevelItem(i)
                assert item is not None
                if item.childCount() > 0:
                    self._observations_tree_widget.setCurrentItem(item.child(0))
                    break

        elif index in {
            _EnsembleWidgetTabs.PARAMETERS_TAB,
            _EnsembleWidgetTabs.MISFIT_TAB,
        }:
            assert self._ensemble is not None

            df: pl.DataFrame = pl.DataFrame()
            with contextlib.suppress(Exception):
                if index == _EnsembleWidgetTabs.PARAMETERS_TAB:
                    df = self._ensemble.load_scalar_keys(transformed=True)
                else:
                    df = self.get_misfit_df()

            table = (
                self._parameters_table
                if index == _EnsembleWidgetTabs.PARAMETERS_TAB
                else self._misfit_table
            )

            table.setUpdatesEnabled(False)
            table.setSortingEnabled(False)
            table.setRowCount(df.height)
            table.setColumnCount(df.width)
            table.setHorizontalHeaderLabels(df.columns)

            rows = df.rows()
            for r, row in enumerate(rows):
                for c, v in enumerate(row):
                    table.setItem(r, c, QTableWidgetItem("" if v is None else str(v)))

            table.resizeColumnsToContents()
            table.setUpdatesEnabled(True)

    def get_misfit_df(self) -> DataFrame:
        assert self._ensemble is not None
        with capture_specific_warning(PerformanceWarning):
            df = LibresFacade.load_all_misfit_data(self._ensemble)
        realization_column = pl.Series(df.index)
        df = pl.from_pandas(df)
        df.insert_column(0, realization_column)
        return df

    @Slot(Ensemble)
    def setEnsemble(self, ensemble: Ensemble) -> None:
        self._ensemble = ensemble

        self._name_label.setText(f"Name: {ensemble.name!s}")
        self._uuid_label.setText(f"UUID: {ensemble.id!s}")

        self._tab_widget.setCurrentIndex(0)

    @Slot()
    def onClickExportMisfit(self) -> None:
        assert self._ensemble is not None
        misfit_df = self.get_misfit_df()
        export_dialog = ExportDialog(misfit_df, "Export misfit", parent=self)
        export_dialog.show()

    @Slot()
    def onClickExportParameters(self) -> None:
        assert self._ensemble is not None
        parameters_df = self._ensemble.load_scalar_keys(transformed=True)
        export_dialog = ExportDialog(
            parameters_df, window_title="Export parameters", parent=self
        )
        export_dialog.show()


class _RealizationWidget(QWidget):
    def __init__(self) -> None:
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

        tab_widget = QTabWidget()
        tab_widget.addTab(info_frame, "Realization")

        layout = QVBoxLayout()
        layout.addWidget(tab_widget)

        self.setLayout(layout)

    @Slot(RealizationStorageState)
    def setRealization(self, ensemble: Ensemble, realization: int) -> None:
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
    def __init__(self) -> None:
        QWidget.__init__(self)

        self._experiment_widget = _ExperimentWidget()
        self._ensemble_widget = _EnsembleWidget()
        self._realization_widget = _RealizationWidget()
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
