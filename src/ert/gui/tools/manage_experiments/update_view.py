from __future__ import annotations

import logging
from typing import cast

from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ert.analysis.event import DataSection
from ert.gui.experiments.view.update import (
    ReportLogTable,
    UpdateLogTable,
    create_table_tab,
)
from ert.storage import Ensemble
from ert.storage.blob_data import StoredUpdate, UpdateStatus, UpdateTable

logger = logging.getLogger(__name__)

_REPORT_TABLE_COLUMNS = {"status", "missing_realizations"}


def _as_data_section(table: UpdateTable) -> DataSection:
    return DataSection(
        header=list(table.header),
        data=[cast(list[str | float], list(row)) for row in table.rows],
        extra=dict(table.summary),
    )


def _table_type(table: UpdateTable) -> type[UpdateLogTable]:
    # ReportLogTable raises unless both columns are present, so an unfinished
    # report from a failed update is shown as a plain table instead.
    if table.is_report and _REPORT_TABLE_COLUMNS.issubset(table.header):
        return ReportLogTable
    return UpdateLogTable


class UpdateView(QWidget):
    """Shows the update that an ensemble produced, as it was recorded in storage.

    An ensemble can be the prior of several updates, for instance when the same
    ensemble has been used to start runs in different experiments. The target
    selector picks which of those updates to show.
    """

    def __init__(self) -> None:
        super().__init__()

        self._posteriors: list[Ensemble] = []

        self._target_selector = QComboBox()
        self._target_selector.setObjectName("update_target_selector")
        self._target_selector.currentIndexChanged.connect(self._show_selected_update)

        self._status_label = QLabel()
        self._status_label.setObjectName("update_status_label")
        self._status_label.setWordWrap(True)

        self._tab_widget = QTabWidget()
        self._tab_widget.setObjectName("stored_update_tabs")

        layout = QVBoxLayout()
        layout.addWidget(self._target_selector)
        layout.addWidget(self._status_label)
        layout.addWidget(self._tab_widget)
        self.setLayout(layout)

    def set_ensemble(self, ensemble: Ensemble) -> None:
        """Find the updates started from this ensemble without reading their tables."""
        try:
            self._posteriors = sorted(
                (child for child in ensemble.children if child.has_stored_update),
                key=lambda child: child.started_at,
            )
        except Exception:
            logger.exception(
                "Could not look for stored updates of ensemble %s", ensemble.name
            )
            self._posteriors = []

        self._clear_tabs()
        self._status_label.clear()

        self._target_selector.blockSignals(True)
        self._target_selector.clear()
        for posterior in self._posteriors:
            self._target_selector.addItem(
                f"{posterior.experiment.name} / {posterior.name}"
            )
        self._target_selector.blockSignals(False)
        self._target_selector.setVisible(len(self._posteriors) > 1)

    @property
    def has_update(self) -> bool:
        return bool(self._posteriors)

    def load_update(self) -> None:
        self._show_selected_update()

    def _clear_tabs(self) -> None:
        # QTabWidget.clear() only removes the tabs, it does not delete their pages.
        while self._tab_widget.count():
            page = self._tab_widget.widget(0)
            self._tab_widget.removeTab(0)
            assert page is not None
            page.setParent(None)
            page.deleteLater()

    def _show_selected_update(self) -> None:
        self._clear_tabs()
        self._status_label.clear()

        index = max(self._target_selector.currentIndex(), 0)
        if index >= len(self._posteriors):
            return

        posterior = self._posteriors[index]
        try:
            update = posterior.load_stored_update()
        except Exception:
            logger.exception("Could not read the stored update of %s", posterior.name)
            self._status_label.setText(
                f"The update that produced {posterior.name} could not be read "
                f"from storage."
            )
            return

        if update is None:
            return

        self._status_label.setText(_describe(update))
        for table in update.tables:
            if not table.header:
                continue
            self._tab_widget.addTab(
                create_table_tab(
                    table.name, _as_data_section(table), _table_type(table)
                ),
                table.name,
            )


def _describe(update: StoredUpdate) -> str:
    if update.status is UpdateStatus.FAILED:
        return (
            f"The {update.update_algorithm} update failed: "
            f"{update.error_message or 'no error message was recorded'}"
        )
    return f"Updated with {update.update_algorithm}"
