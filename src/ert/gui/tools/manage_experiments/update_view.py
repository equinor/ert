from __future__ import annotations

import io
import logging

import polars as pl
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
from ert.storage.blob_data import (
    BlobStorageData,
    BlobType,
    ObservationReportData,
    UpdateDataTableData,
    UpdateStatus,
)

logger = logging.getLogger(__name__)

_REPORT_TABLE_COLUMNS = {"status", "missing_realizations"}


def _table_index(blob: BlobStorageData) -> int:
    assert isinstance(blob.blob_info, UpdateDataTableData)
    return blob.blob_info.table_index


def _read_data_section(
    ensemble: Ensemble, uri: str, summary: dict[str, str]
) -> DataSection:
    table = pl.read_parquet(io.BytesIO(ensemble.load_blob(uri)))
    return DataSection(header=table.columns, data=table.rows(), extra=summary)


def _report_table_type(report: DataSection) -> type[UpdateLogTable]:
    # ReportLogTable raises unless both columns are present, so an unfinished
    # report from a failed update is shown as a plain table instead.
    if _REPORT_TABLE_COLUMNS.issubset(report.header):
        return ReportLogTable
    return UpdateLogTable


def _has_readable_update(ensemble: Ensemble) -> bool:
    try:
        return bool(ensemble.load_blob_metadata(BlobType.OBSERVATION_REPORT))
    except Exception:
        logger.exception("Could not read the blob metadata of ensemble %s", ensemble.id)
        return False


class UpdateView(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._ensemble: Ensemble | None = None
        self._has_update = False

        self._description_label = QLabel()
        self._description_label.setWordWrap(True)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)

        self._tab_widget = QTabWidget()

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
                f"Update {update_iteration}: iter-{update_iteration}"
                f" \u2192 iter-{ensemble.iteration}"
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
            self._show_update(self._ensemble)
        except Exception:
            logger.exception(
                "Could not read the stored update of %s", self._ensemble.name
            )
            self._clear_tabs()
            self._status_label.setText(
                f"The update that produced {self._ensemble.name} could not be read "
                f"from storage."
            )

    def _show_update(self, ensemble: Ensemble) -> None:
        [report_blob] = ensemble.load_blob_metadata(BlobType.OBSERVATION_REPORT)
        report_info = report_blob.blob_info
        assert isinstance(report_info, ObservationReportData)
        self._status_label.setText(_describe(report_info))

        table_blobs = ensemble.load_blob_metadata(BlobType.UPDATE_DATA_TABLE)
        for table_blob in sorted(table_blobs, key=_table_index):
            table_info = table_blob.blob_info
            assert isinstance(table_info, UpdateDataTableData)
            table = _read_data_section(ensemble, table_blob.uri, table_info.summary)
            self._add_tab(table_info.table_name, table, UpdateLogTable)

        report = _read_data_section(ensemble, report_blob.uri, report_info.summary)
        self._add_tab("Report", report, _report_table_type(report))

    def _add_tab(
        self, name: str, data: DataSection, table_type: type[UpdateLogTable]
    ) -> None:
        if not data.header:
            return
        self._tab_widget.addTab(create_table_tab(name, data, table_type), name)

    def _clear_tabs(self) -> None:
        # QTabWidget.clear() only removes the tabs, it does not delete their pages.
        while self._tab_widget.count():
            page = self._tab_widget.widget(0)
            self._tab_widget.removeTab(0)
            if page is not None:
                page.deleteLater()


def _describe(report: ObservationReportData) -> str:
    if report.status is UpdateStatus.FAILED:
        return (
            f"The {report.update_algorithm} update failed: "
            f"{report.error_message or 'no error message was recorded'}"
        )
    return f"Updated with {report.update_algorithm}"
