import os
from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QMessageBox,
)

from ert.gui.ertwidgets.listeditbox import ListEditBox
from ert.gui.ertwidgets.pathchooser import PathChooser
from ert.gui.tools.export.export_panel import ExportDialog
from ert.libres_facade import LibresFacade

from .conftest import (
    get_child,
    wait_for_child,
)


def test_csv_export(esmda_has_run, qtbot):
    gui = esmda_has_run

    file_name = None

    def handle_export_dialog():
        nonlocal file_name

        # Find the case selection box in the dialog
        export_dialog = wait_for_child(gui, qtbot, ExportDialog)
        case_selection = get_child(export_dialog, ListEditBox)

        # Select default_0 as the case to be exported
        case_selection._list_edit_line.setText("default_0")
        path_chooser = get_child(export_dialog, PathChooser)
        file_name = path_chooser._path_line.text()
        assert case_selection.isValid()

        qtbot.mouseClick(export_dialog.ok_button, Qt.MouseButton.LeftButton)

    def handle_finished_box():
        """
        Click on the plugin finished dialog once it pops up
        """
        finished_message = wait_for_child(gui, qtbot, QMessageBox)
        assert "completed" in finished_message.text()
        qtbot.mouseClick(
            finished_message.button(QMessageBox.Ok), Qt.MouseButton.LeftButton
        )

    QTimer.singleShot(500, handle_export_dialog)
    QTimer.singleShot(3000, handle_finished_box)

    gui.tools["Export data"].trigger()
    assert file_name == "output.csv"
    qtbot.waitUntil(lambda: os.path.exists(file_name))

    file_content = Path(file_name).read_text(encoding="utf-8")
    ensemble = gui.notifier.storage.get_ensemble_by_name("default_0")
    gen_kw_data = ensemble.load_all_gen_kw_data()

    facade = LibresFacade.from_config_file("poly.ert")
    misfit_data = facade.load_all_misfit_data(ensemble)

    for i in range(ensemble.ensemble_size):
        assert (
            f"{i},0,,default_0,{gen_kw_data.iloc[i]['COEFFS:a']},{gen_kw_data.iloc[i]['COEFFS:b']},{gen_kw_data.iloc[i]['COEFFS:c']},{misfit_data.iloc[i]['MISFIT:POLY_OBS']},{misfit_data.iloc[i]['MISFIT:TOTAL']}"
            in file_content
        )
