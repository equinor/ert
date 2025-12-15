from PyQt6.QtWidgets import QHBoxLayout

from ert.gui.tools.manage_experiments.storage_info_widget import ExportMisfitWidget


def test_get_selected_file_extension_returns_text_of_active_radio_button(qtbot):
    widget = ExportMisfitWidget()

    assert widget.get_selected_file_extension() == "csv"

    hdf_button = widget._export_format_section.findChild(QHBoxLayout).itemAt(1).widget()
    assert hdf_button.text() == "hdf"
    hdf_button.click()

    assert widget.get_selected_file_extension() == "hdf"


def test_export_misfit_widget_has_selected_a_radio_button_by_default(qtbot):
    widget = ExportMisfitWidget()
    button_layout = widget._export_format_section.findChild(QHBoxLayout)
    assert any(
        button_layout.itemAt(i).widget()
        for i in range(button_layout.count())
        if button_layout.itemAt(i).widget().isChecked()
    )
