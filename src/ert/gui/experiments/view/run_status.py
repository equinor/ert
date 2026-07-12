import logging
from pathlib import Path
from typing import cast

from PyQt6.QtCore import QModelIndex, Qt
from PyQt6.QtWidgets import (
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ert.ensemble_evaluator.event import FullSnapshotEvent
from ert.gui.experiments.run_dialog import FMStepOverview
from ert.gui.experiments.view.progress_widget import ProgressWidget
from ert.gui.experiments.view.realization import RealizationWidget
from ert.gui.model.real_list import RealListModel
from ert.gui.model.snapshot import SnapshotModel
from ert.run_models.event import (
    CorruptStatusSnapshotError,
    load_status_snapshot_event,
)

logger = logging.getLogger(__name__)


class RunStatusView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._stack = QStackedWidget(self)

        self._placeholder = QLabel(
            "No run status snapshot is available for this ensemble.", self
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(self._placeholder)

        self._error_label = QLabel(self)
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(self._error_label)

        self._content: QWidget | None = None
        self._loaded_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)

    def load_snapshot(self, path: Path) -> None:
        if path == self._loaded_path:
            return
        self._loaded_path = path
        if self._content is not None:
            self._stack.removeWidget(self._content)
            self._content.deleteLater()
            self._content = None

        try:
            event = load_status_snapshot_event(path)
        except CorruptStatusSnapshotError:
            logger.warning(f"Corrupt status snapshot at {path}", exc_info=True)
            self._error_label.setText(
                "The run status snapshot for this ensemble could not be read "
                "and may be corrupted."
            )
            self._stack.setCurrentWidget(self._error_label)
            return
        if event is None or event.snapshot is None:
            self._stack.setCurrentWidget(self._placeholder)
            return

        SnapshotModel.prerender(event.snapshot)
        self._content = self._build_content(event)
        self._stack.addWidget(self._content)
        self._stack.setCurrentWidget(self._content)

    def _build_content(self, event: FullSnapshotEvent) -> QWidget:
        content = QWidget(self)
        snapshot_model = SnapshotModel(content)
        snapshot = event.snapshot
        assert snapshot is not None
        snapshot_model._add_snapshot(snapshot, str(event.iteration))

        fm_step_overview = FMStepOverview(snapshot_model, self)
        fm_step_label = QLabel(self)

        realization_widget = RealizationWidget(0)
        realization_widget.setSnapshotModel(snapshot_model)

        def select_real(index: QModelIndex) -> None:
            if not index.isValid():
                return
            iter_ = cast(RealListModel, index.model()).get_iter()
            fm_step_overview.set_realization(iter_, index.row())
            fm_step_label.setText(
                f"Realization id {index.row()} in iteration {event.iteration}"
            )

        realization_widget.itemClicked.connect(select_real)

        progress_widget = ProgressWidget()
        progress_widget.update_progress(event.status_count, event.realization_count)

        content_layout = QVBoxLayout(content)
        content_layout.addWidget(progress_widget)
        content_layout.addWidget(realization_widget)
        content_layout.addWidget(fm_step_label)
        content_layout.addWidget(fm_step_overview)

        first_real = realization_widget._real_list_model.index(0, 0)
        if first_real.isValid():
            select_real(first_real)

        return content
