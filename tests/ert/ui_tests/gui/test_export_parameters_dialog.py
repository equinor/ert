import pathlib
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from PyQt6.QtGui import QColor

from ert.gui.tools.manage_experiments.export_parameters_dialog import (
    ExportParametersDialog,
)
from ert.storage import Storage


def test_that_export_writes_to_file(
    qtbot, snake_oil_storage: Storage, change_to_tmpdir
):
    ensemble = next(snake_oil_storage.ensembles)
    dialog = ExportParametersDialog(ensemble)
    qtbot.addWidget(dialog)
    dialog.show()

    assert not dialog._file_path_edit.text()
    assert dialog._export_button.isEnabled() is False
    assert not dialog._export_text_area.toPlainText()

    dialog._file_path_edit.setText("test_export.csv")
    assert dialog._export_button.isEnabled() is True

    dialog.export()
    assert (
        "Ensemble parameters exported to: test_export.csv"
        in dialog._export_text_area.toPlainText()
    )

    assert_frame_equal(
        ensemble.load_all_scalar_keys(transformed=True),
        pl.read_csv("test_export.csv"),
        abs_tol=1e-6,
    )


@pytest.mark.parametrize(
    "invalid_path",
    [
        "",
        "/non/existent/path/export.csv",
        "   ",
        "hhhee/\0invalid.csv",
        "/",
        "\\",
    ],
)
def test_file_path_validation_invalid(qtbot, snake_oil_storage: Storage, invalid_path):
    ensemble = next(snake_oil_storage.ensembles)
    dialog = ExportParametersDialog(ensemble)
    qtbot.addWidget(dialog)
    dialog.show()

    dialog._file_path_edit.setText(invalid_path)
    dialog.validate_file()

    assert dialog._export_button.isEnabled() is False
    palette = dialog._file_path_edit.palette()
    assert palette.color(palette.ColorRole.Text) == QColor("red")
    assert dialog._file_path_edit.toolTip() == "Invalid file path"


@pytest.mark.parametrize(
    "valid_path",
    [
        "valid_export.csv",
        "   valid_export.csv   ",
        "subdir/valid_export.csv",
    ],
)
def test_file_path_validation_valid(
    qtbot, snake_oil_storage: Storage, valid_path, change_to_tmpdir
):
    ensemble = next(snake_oil_storage.ensembles)
    dialog = ExportParametersDialog(ensemble)
    qtbot.addWidget(dialog)
    dialog.show()

    pathlib.Path("subdir").mkdir()
    dialog._file_path_edit.setText(valid_path)
    dialog.validate_file()

    assert dialog._export_button.isEnabled() is True
    palette = dialog._file_path_edit.palette()
    assert palette.color(palette.ColorRole.Text) == QColor("black")
    assert not dialog._file_path_edit.toolTip()


@patch("ert.storage.Ensemble.load_all_scalar_keys")
def test_export_failure_handling(
    patched_load_all_scalar_keys, qtbot, snake_oil_storage: Storage
):
    patched_load_all_scalar_keys.side_effect = Exception("i_am_an_exception")

    ensemble = next(snake_oil_storage.ensembles)
    dialog = ExportParametersDialog(ensemble)
    qtbot.addWidget(dialog)
    dialog.show()

    dialog._file_path_edit.setText("test_export.csv")
    dialog.export()
    assert (
        "Error exporting ensemble parameters: i_am_an_exception"
        in dialog._export_text_area.toPlainText()
    )
