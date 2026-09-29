from __future__ import annotations

import logging
from typing import cast

from PyQt6.QtWidgets import (
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


def _has_readable_update(ensemble: Ensemble) -> bool:
    try:
        return ensemble.has_stored_update
    except Exception:
        logger.exception("Could not read the blob metadata of ensemble %s", ensemble.id)
        return False


class UpdateView(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._ensemble: Ensemble | None = None
        self._has_update = False

        self._description_label = QLabel()
        self._description_label.setObjectName("update_description_label")
        self._description_label.setWordWrap(True)

        self._status_label = QLabel()
        self._status_label.setObjectName("update_status_label")
        self._status_label.setWordWrap(True)

        self._tab_widget = QTabWidget()
        self._tab_widget.setObjectName("stored_update_tabs")

        layout = QVBoxLayout()
        layout.addWidget(self._description_label)
        layout.addWidget(self._status_label)
        layout.addWidget(self._tab_widget)
        self.setLayout(layout)

    def set_ensemble(self, ensemble: Ensemble) -> None:
        """Check for an update producing this ensemble without reading its tables."""
        self._ensemble = ensemble
        self._has_update = _has_readable_update(ensemble)

        self._clear_tabs()
        self._description_label.clear()
        self._status_label.clear()

        if self.has_update:
            update_iteration = ensemble.iteration - 1
            self._description_label.setText(
                f"Update {update_iteration}: input iteration {update_iteration}"
                f" \u2192 output iteration {ensemble.iteration}"
            )

    @property
    def has_update(self) -> bool:
        return self._has_update

    def load_update(self) -> None:
        self._clear_tabs()
        self._status_label.clear()

        if not self.has_update:
            return

        assert self._ensemble is not None
        try:
            update = self._ensemble.load_stored_update()
        except Exception:
            logger.exception(
                "Could not read the stored update of %s", self._ensemble.name
            )
            self._status_label.setText(
                f"The update that produced {self._ensemble.name} could not be read "
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

    def _clear_tabs(self) -> None:
        # QTabWidget.clear() only removes the tabs, it does not delete their pages.
        while self._tab_widget.count():
            page = self._tab_widget.widget(0)
            self._tab_widget.removeTab(0)
            assert page is not None
            page.setParent(None)
            page.deleteLater()


def _describe(update: StoredUpdate) -> str:
    if update.status is UpdateStatus.FAILED:
        return (
            f"The {update.update_algorithm} update failed: "
            f"{update.error_message or 'no error message was recorded'}"
        )
    return f"Updated with {update.update_algorithm}"
