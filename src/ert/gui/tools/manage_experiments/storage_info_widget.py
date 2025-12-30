import contextlib
import json
from collections.abc import Callable
from enum import IntEnum
from pathlib import Path

import polars as pl
import seaborn as sns
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvas  # type: ignore
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtCore import pyqtSlot as Slot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
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

from ert.plugins.hook_implementations.workflows import ExportMisfitDataJob
from ert.storage import Ensemble, Experiment, RealizationStorageState

from .export_parameters_dialog import ExportParametersDialog


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
    EXPORT_MISFIT_TAB = 4


class _EnsembleWidgetTabs(IntEnum):
    ENSEMBLE_TAB = 0
    STATE_TAB = 1
    OBSERVATIONS_TAB = 2
    PARAMETERS_TAB = 3
    EXPORT_MISFIT_TAB = 4


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
        self._export_misfit_widget = ExportMisfitWidget()
        tab_widget.insertTab(
            _ExperimentWidgetTabs.EXPORT_MISFIT_TAB,
            self._export_misfit_widget,
            "Export Misfit",
        )

        layout = QVBoxLayout()
        layout.addWidget(tab_widget)

        self.setLayout(layout)

    @Slot(Experiment)
    def setExperiment(self, experiment: Experiment) -> None:
        self._experiment = experiment
        self._export_misfit_widget.set_experiment(experiment)

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

        parameters_frame = QFrame()
        parameters_layout = QVBoxLayout()
        self._parameters_table = QTableWidget()
        vertical_header = self._parameters_table.verticalHeader()
        assert vertical_header is not None
        vertical_header.setVisible(False)
        parameters_layout.addWidget(self._parameters_table)
        self._export_button = QPushButton("Export...")
        self._export_button.clicked.connect(self.onClickExportParameters)
        parameters_layout.addWidget(self._export_button)
        parameters_frame.setLayout(parameters_layout)

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
            _ExperimentWidgetTabs.EXPORT_MISFIT_TAB,
            QWidget(),
            "Export Misfit",
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

        observation_key = selected.data(1, Qt.ItemDataRole.DisplayRole)
        if not observation_key:
            return

        observation_label = selected.data(0, Qt.ItemDataRole.DisplayRole)
        assert self._ensemble is not None
        observations_dict = self._ensemble.experiment.observations

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_title(observation_key)
        ax.grid(True)

        response_type, obs_for_type = next(
            (
                (response_type, df)
                for response_type, df in observations_dict.items()
                if observation_key in df["observation_key"]
            ),
            (None, None),
        )

        assert response_type is not None
        assert obs_for_type is not None

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
                    return_dtype=pl.String,
                )
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
                        response_key, Qt.MatchFlag.MatchExactly
                    )
                    if len(match_list) == 0:
                        root = QTreeWidgetItem(
                            self._observations_tree_widget, [response_key]
                        )
                    else:
                        root = match_list[0]

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

                    self._observations_tree_widget.sortItems(
                        0, Qt.SortOrder.AscendingOrder
                    )

            for i in range(self._observations_tree_widget.topLevelItemCount()):
                item = self._observations_tree_widget.topLevelItem(i)
                assert item is not None
                if item.childCount() > 0:
                    self._observations_tree_widget.setCurrentItem(item.child(0))
                    break

        elif index == _EnsembleWidgetTabs.PARAMETERS_TAB:
            assert self._ensemble is not None

            df: pl.DataFrame = pl.DataFrame()
            with contextlib.suppress(Exception):
                df = self._ensemble.load_scalar_keys(transformed=True)

            table = self._parameters_table

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

    @Slot(Ensemble)
    def setEnsemble(self, ensemble: Ensemble) -> None:
        self._ensemble = ensemble

        self._name_label.setText(f"Name: {ensemble.name!s}")
        self._uuid_label.setText(f"UUID: {ensemble.id!s}")

        self._tab_widget.setCurrentIndex(0)

    @Slot()
    def onClickExportParameters(self) -> None:
        assert self._ensemble is not None
        export_dialog = ExportParametersDialog(self._ensemble, self)
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


def group_radio_buttons(layout: QLayout) -> QButtonGroup:
    format_button_group: QButtonGroup = QButtonGroup()

    for i in range(layout.count()):
        item = layout.itemAt(i)

        if item is not None:
            widget = item.widget()
            if isinstance(widget, QRadioButton):
                format_button_group.addButton(widget)
    return format_button_group


class SectionTitle(QLabel):
    def __init__(self, text: str, font_size: int = 14) -> None:
        QLabel.__init__(self, text)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPixelSize(font_size)
        self.setFont(title_font)


class ExportFormatSection(QVBoxLayout):
    def __init__(self) -> None:
        QVBoxLayout.__init__(self)
        self.addWidget(SectionTitle("Export format"))
        button_layout = QHBoxLayout()
        csv_button = QRadioButton("csv")
        csv_button.setChecked(True)
        button_layout.addWidget(csv_button)
        button_layout.addWidget(QRadioButton("hdf"))
        self.addLayout(button_layout)


class FileSection(QVBoxLayout):
    LINE_WIDTH = 200
    DEFAULT_FILE_EXTENSION = ".csv"

    def __init__(self, save_action: Callable[[str], None]) -> None:
        super().__init__()
        self.addWidget(SectionTitle("Save to file"))

        text_layout = QHBoxLayout()
        text_layout.addWidget(QLabel("File name:"))
        self.text_field = QLineEdit()
        self.text_field.setFixedWidth(self.LINE_WIDTH)
        self.text_field.setText("misfit_data")
        text_layout.addWidget(self.text_field)

        self.file_extension = QLabel(self.DEFAULT_FILE_EXTENSION)
        text_layout.addWidget(self.file_extension)

        self.save_button = QPushButton("Save")
        self.save_button.pressed.connect(lambda: save_action(self.text_field.text()))
        text_layout.addWidget(self.save_button)

        self.addLayout(text_layout)


class ExportMisfitWidget(QWidget):
    def __init__(self) -> None:
        QWidget.__init__(self)
        self._experiment: Experiment | None = None

        main_layout = QVBoxLayout()
        main_layout.addStretch()
        main_layout.addWidget(SectionTitle("Misfit", font_size=20))

        self._export_format_section = ExportFormatSection()
        main_layout.addLayout(self._export_format_section)

        self._file_format_button_group: QButtonGroup = group_radio_buttons(
            self._export_format_section.findChild(QHBoxLayout)
        )

        self._file_section: FileSection = FileSection(self.save_file)
        self._file_format_button_group.buttonClicked.connect(
            lambda: self._file_section.file_extension.setText(
                f".{self.get_selected_file_extension()}"
            )
        )
        self._file_format_button_group.buttonClicked.connect(self.validate_filename)
        self._file_section.text_field.textChanged.connect(self.validate_filename)
        self._file_section.save_button.clicked.connect(self.validate_filename)
        self.validate_filename()

        main_layout.addLayout(self._file_section)

        main_layout.setSpacing(5)

        self.recursively_style_layout(main_layout)
        self.setLayout(main_layout)

    def set_experiment(self, experiment: Experiment) -> None:
        self._experiment = experiment

    def get_file_name(self) -> str:
        return (
            f"{self._file_section.text_field.text()}"
            f"."
            f"{self.get_selected_file_extension()}"
        )

    def validate_filename(self) -> None:
        file_name = self.get_file_name()
        if Path(file_name).is_file():
            self._file_section.save_button.setDisabled(True)
            self._file_section.save_button.setToolTip(f"'{file_name}' already exists!")
        else:
            self._file_section.save_button.setDisabled(False)
            self._file_section.save_button.setToolTip("")

    def recursively_style_layout(self, layout: QBoxLayout) -> None:
        """Recursively set the style of the layout, styling children
        QLayouts and QWidgets"""
        for widget in layout.findChildren(QWidget):
            widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        for child_layout in layout.findChildren(QBoxLayout):
            self.recursively_style_layout(child_layout)
        layout.setSpacing(5)
        if isinstance(layout, QHBoxLayout):
            layout.addStretch()

    def get_selected_file_extension(self) -> str:
        if checked_button := self._file_format_button_group.checkedButton():
            return checked_button.text()
        return ""

    def save_file(self, file_name: str) -> None:
        if file_name is None or self._experiment is None:
            return

        selected_file_format = self.get_selected_file_extension()

        file_name = f"{file_name}.{selected_file_format}"
        for ensemble in self._experiment.ensembles:
            ExportMisfitDataJob().run(ensemble, workflow_args=[file_name])
        print(f"Saved to {file_name}!")
