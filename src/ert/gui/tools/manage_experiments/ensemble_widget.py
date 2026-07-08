import contextlib
import io
from enum import IntEnum
from typing import cast

import polars as pl
import seaborn as sns
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
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ert.config.response_config import BaseResponseConfig
from ert.config.rft_config import RFTConfig
from ert.gui.experiments.view.run_status import RunStatusView
from ert.storage import Ensemble
from ert.storage.blob_data import BlobType
from ert.warnings import capture_specific_warning

from .export_dialog import ExportDialog


class _EnsembleWidgetTabs(IntEnum):
    ENSEMBLE_TAB = 0
    STATE_TAB = 1
    OBSERVATIONS_TAB = 2
    PARAMETERS_TAB = 3
    MISFIT_TAB = 4
    RUN_STATUS_TAB = 5


class _ObservationTreeWidgetItem(QTreeWidgetItem):
    def __init__(
        self,
        parent: QTreeWidgetItem,
        observation_key: str,
        observation_data: pl.DataFrame,
        response_config: BaseResponseConfig,
    ) -> None:
        assert len(observation_data) == 1
        self.observation_key = observation_key
        self.observation_data = observation_data
        self.response_config = response_config
        super().__init__(parent, [self.display_label, observation_key])

    def match_key_data(self, metadata: pl.DataFrame | None) -> dict[str, object]:
        response_type = self.response_config.type
        if response_type == "rft":
            assert metadata is not None
            observation_data = RFTConfig.enrich_observations_with_metadata(
                self.observation_data, metadata
            )
            observation_data = observation_data.filter(RFTConfig.is_zone_valid())
        else:
            observation_data = self.observation_data

        if observation_data.is_empty():
            return {}
        return observation_data.select(self.response_config.match_key).row(
            0, named=True
        )

    @property
    def index_key_data(self) -> dict[str, object]:
        return self.observation_data.select(self.response_config.index_key).row(
            0, named=True
        )

    @property
    def display_label(self) -> str:
        return ", ".join(
            self.response_config.display_column(val, col)
            for col, val in self.index_key_data.items()
        )

    def __lt__(self, other: QTreeWidgetItem) -> bool:
        assert isinstance(other, _ObservationTreeWidgetItem)
        assert self.response_config.index_key == other.response_config.index_key, (
            "Expecting items being compared to be of the same response type"
        )

        for val, other_val in zip(
            self.index_key_data.values(),
            other.index_key_data.values(),
            strict=True,
        ):
            if isinstance(val, (int, float)) and isinstance(other_val, (int, float)):
                if val != other_val:
                    return val < other_val
                continue

            val_str = str(val)
            other_val_str = str(other_val)
            if val_str != other_val_str:
                return val_str < other_val_str

        return False


class EnsembleWidget(QWidget):
    def __init__(self) -> None:
        QWidget.__init__(self)
        self._ensemble: Ensemble | None = None

        info_frame = QFrame()
        self._name_label = QLabel()
        self._uuid_label = QLabel()
        self._iteration_label = QLabel()

        info_layout = QVBoxLayout()
        info_layout.addWidget(self._name_label)
        info_layout.addWidget(self._iteration_label)
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

        self._run_status_view = RunStatusView()

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
        self._tab_widget.insertTab(
            _EnsembleWidgetTabs.RUN_STATUS_TAB, self._run_status_view, "Run status"
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
        if not selected or not isinstance(selected, _ObservationTreeWidgetItem):
            return

        assert self._ensemble is not None

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        ax.set_title(selected.observation_key)
        ax.grid(True)
        obs = selected.observation_data
        scaling_blobs = self._ensemble.load_blobs(BlobType.SCALING_FACTORS)
        if scaling_blobs:
            try:
                raw = self._ensemble.load_blob(scaling_blobs[0].uri)
                scaling_df = pl.read_parquet(io.BytesIO(raw))
            except FileNotFoundError:
                scaling_df = None
        else:
            scaling_df = None

        def _try_render_scaled_obs() -> None:
            if scaling_df is None:
                return

            index_col = selected.response_config.index_column_expr()
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

        response_type = selected.response_config.type
        response_key = str(obs["response_key"].item())
        reals_with_responses = tuple(
            self._ensemble.get_realization_list_with_responses()
        )

        def _filter_by_match_key(
            df: pl.DataFrame, observation_metadata: pl.DataFrame | None
        ) -> pl.DataFrame:
            """Filter df to match 'selected' observation on match keys."""
            match_data = selected.match_key_data(observation_metadata)
            if not match_data:
                return df.filter(pl.lit(False))

            mask = pl.lit(True)
            for col, val in match_data.items():
                mask &= pl.col(col).eq_missing(val)
            return df.filter(mask)

        def _load_and_filter_responses() -> pl.DataFrame | None:
            assert self._ensemble is not None
            ens = self._ensemble

            if not reals_with_responses:
                return None

            if response_key not in ens.experiment.response_key_to_response_type:
                return None

            if response_type == "rft":
                per_realization: list[pl.DataFrame] = []
                for real in reals_with_responses:
                    response = ens.load_responses(response_key, (real,))
                    observation_metadata = ens.load_observation_location_metadata(real)
                    if cast(
                        RFTConfig, selected.response_config
                    ).approximate_missing_values:
                        response = RFTConfig.approximate_missing_rft_responses(
                            response.lazy(),
                            RFTConfig.enrich_observations_with_metadata(
                                obs, observation_metadata
                            ),
                        ).collect()
                    res = _filter_by_match_key(response, observation_metadata)
                    per_realization.append(res)
                return pl.concat(per_realization)

            responses = ens.load_responses(response_key, reals_with_responses)
            return _filter_by_match_key(responses, None)

        response_ds = _load_and_filter_responses()

        if response_ds is not None and not response_ds.is_empty():
            response_ds_for_label = response_ds.rename({"values": "Responses"})[
                ["response_key", "Responses"]
            ]

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
                configs = exp.base_response_configuration
                if response_type not in configs:
                    continue
                response_config = configs[response_type]

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
                    ).unique()

                    for observation_data in obs_ds.iter_slices(n_rows=1):
                        _ObservationTreeWidgetItem(
                            root, obs_key, observation_data, response_config
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

        elif index == _EnsembleWidgetTabs.RUN_STATUS_TAB:
            assert self._ensemble is not None
            self._run_status_view.load_snapshot(
                self._ensemble.experiment.status_snapshot_path(self._ensemble.iteration)
            )

    def get_misfit_df(self) -> DataFrame:
        assert self._ensemble is not None
        with capture_specific_warning(PerformanceWarning):
            df = self._ensemble.load_all_misfit_data()
        realization_column = pl.Series(df.index)
        df = pl.from_pandas(df)
        df.insert_column(0, realization_column)
        return df

    @Slot(Ensemble)
    def setEnsemble(self, ensemble: Ensemble) -> None:
        self._ensemble = ensemble

        self._name_label.setText(f"Name: {ensemble.name!s}")
        self._uuid_label.setText(f"UUID: {ensemble.id!s}")
        self._iteration_label.setText(f"Iteration: {ensemble.iteration:d}")

        current_index = self._tab_widget.currentIndex()
        if current_index > 0:
            self._currentTabChanged(current_index)
        else:
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
