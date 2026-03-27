from uuid import uuid4

import numpy as np
import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QMessageBox, QTableWidget
from pytestqt.qtbot import QtBot

from ert.analysis.event import DataSection
from ert.analysis.snapshots import ObservationStatus
from ert.gui.experiments.view import UpdateWidget
from ert.gui.experiments.view.update import ReportLogTable
from ert.run_models.event import RunModelDataEvent


def test_update_widget(qtbot: QtBot):
    event = RunModelDataEvent(
        iteration=0,
        name="test",
        run_id=uuid4(),
        data=DataSection(header=["a", "b"], data=np.array([[42, 2], [3, 4]])),
    )
    widget = UpdateWidget(event.iteration)
    widget.show()
    qtbot.addWidget(widget)
    widget.add_table(event)
    table = widget.findChild(QTableWidget, "CSV_test")

    assert table is not None
    assert (table.columnCount(), table.rowCount()) == (2, 2)
    assert table.item(1, 1).text() == "4"


@pytest.mark.parametrize(
    ("existing_columns", "missing_column"),
    [
        pytest.param(["a", "b"], "status", id="status"),
        pytest.param(
            ["a", "status"], "missing_realizations", id="missing_realizations"
        ),
    ],
)
def test_that_report_log_table_fails_without_required_column(
    qtbot: QtBot, existing_columns, missing_column
):
    # qtbot provides QApplication, without which ReportLogTable fails
    with pytest.raises(
        RuntimeError,
        match=f"'{missing_column}' column should be present in the report table",
    ):
        ReportLogTable(DataSection(header=existing_columns, data=[[42, 2], [3, 4]]))


def test_that_report_log_table_does_not_underline_status_other_than_nan(qtbot: QtBot):
    headers = ["missing_realizations", "value", "status"]
    status_column = 2
    observations = [
        ["dummy", 1, ObservationStatus.ACTIVE],
        ["dummy", 2, ObservationStatus.MISSING_RESPONSE],
        ["dummy", 3, ObservationStatus.OUTLIER],
        ["dummy", 4, ObservationStatus.STD_CUTOFF],
    ]

    nan_row = 1

    report_table = ReportLogTable(DataSection(header=headers, data=observations))
    qtbot.addWidget(report_table)

    for row in range(len(observations)):
        for column in range(len(headers)):
            is_underlined = (column == status_column) and (row == nan_row)
            assert report_table.item(row, column).font().underline() == is_underlined, (
                f"({row}, {column}) must {'' if is_underlined else 'not'} be underlined"
            )


def click_on_table_cell(qtbot, report_table, row, column):
    cell_rect = report_table.visualItemRect(report_table.item(row, column))
    qtbot.mouseClick(
        report_table.viewport(), Qt.MouseButton.LeftButton, pos=cell_rect.center()
    )


def verify_disabled_observations_dialog_shows_on_click(
    qtbot, report_table, nan_message, row, column
):
    def handle_disabled_observations_blocking_dialog(qtbot, report_table, nan_message):
        message_box = report_table.findChild(QMessageBox)
        try:
            assert message_box.text() == nan_message
        finally:
            qtbot.mouseClick(
                message_box.button(QMessageBox.StandardButton.Ok),
                Qt.MouseButton.LeftButton,
            )

    QTimer.singleShot(
        500,
        lambda: handle_disabled_observations_blocking_dialog(
            qtbot, report_table, nan_message
        ),
    )

    click_on_table_cell(qtbot, report_table, row, column)


def test_that_report_log_table_only_shows_message_on_nan_status_click(qtbot: QtBot):
    headers = ["status", "value", "missing_realizations"]
    status_column = 0
    data_column = 1
    hidden_column = 2

    observations = [
        [ObservationStatus.ACTIVE, 10, ""],
        [ObservationStatus.MISSING_RESPONSE, 42, "1, 3"],
        [ObservationStatus.OUTLIER, 666, "EVIL"],
        [ObservationStatus.STD_CUTOFF, 66, "5, 6"],
    ]

    nan_rows = [1]
    nan_message = "Missing responses from active realizations: 1, 3"

    report_table = ReportLogTable(DataSection(header=headers, data=observations))
    qtbot.addWidget(report_table)

    def verify_that_status_column_click_shows_dialog_for_missing_observations_only(
        row, column
    ):
        if row in nan_rows:
            verify_disabled_observations_dialog_shows_on_click(
                qtbot, report_table, nan_message, row, column
            )
        else:
            click_on_table_cell(qtbot, report_table, row, column)
            assert report_table.findChild(QMessageBox) is None

    def verify_that_data_column_click_does_not_produce_dialog(row, column):
        click_on_table_cell(qtbot, report_table, row, column)
        assert report_table.findChild(QMessageBox) is None

    assert report_table.horizontalHeader().isSectionHidden(hidden_column)

    for row in range(len(observations)):
        assert report_table.item(row, status_column).text() == observations[row][0]

        verify_that_status_column_click_shows_dialog_for_missing_observations_only(
            row, status_column
        )
        verify_that_data_column_click_does_not_produce_dialog(row, data_column)


def test_that_report_log_table_matches_data_on_sort(qtbot: QtBot):
    headers = ["value", "status", "missing_realizations"]
    value_column = 0
    status_column = 1
    observations = [
        [100, ObservationStatus.MISSING_RESPONSE, "10"],
        [200, ObservationStatus.MISSING_RESPONSE, "11"],
    ]

    report_table = ReportLogTable(DataSection(header=headers, data=observations))
    qtbot.addWidget(report_table)

    row100 = 0
    row200 = 1

    nan_message100 = "Missing responses from active realizations: 10"
    nan_message200 = "Missing responses from active realizations: 11"

    verify_disabled_observations_dialog_shows_on_click(
        qtbot, report_table, nan_message100, row100, status_column
    )
    verify_disabled_observations_dialog_shows_on_click(
        qtbot, report_table, nan_message200, row200, status_column
    )

    report_table.sortItems(value_column, Qt.SortOrder.DescendingOrder)
    row100 = 1
    row200 = 0

    verify_disabled_observations_dialog_shows_on_click(
        qtbot, report_table, nan_message100, row100, status_column
    )

    verify_disabled_observations_dialog_shows_on_click(
        qtbot, report_table, nan_message200, row200, status_column
    )
