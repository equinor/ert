import os
from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QMessageBox,
    QPushButton,
)

from ert.gui.ertwidgets.customdialog import CustomDialog
from ert.gui.ertwidgets.listeditbox import ListEditBox
from ert.gui.ertwidgets.pathchooser import PathChooser
from ert.libres_facade import LibresFacade

from .conftest import (
    get_child,
    wait_for_child,
)


def test_csv_export(esmda_has_run, qtbot):
    gui = esmda_has_run

    # Find EXPORT_CSV in the plugin menu
    plugin_tool = gui.tools["Plugins"]
    plugin_actions = plugin_tool.getAction().menu().actions()
    export_csv_action = [a for a in plugin_actions if a.text() == "CSV Export"][0]

    file_name = None

    def handle_plugin_dialog():
        nonlocal file_name

        # Find the case selection box in the dialog
        export_dialog = wait_for_child(gui, qtbot, CustomDialog)
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

    QTimer.singleShot(500, handle_plugin_dialog)
    QTimer.singleShot(3000, handle_finished_box)
    export_csv_action.trigger()

    runner = plugin_tool.get_plugin_runner("CSV Export")
    assert runner.poll_thread is not None
    if runner.poll_thread.is_alive():
        runner.poll_thread.join()

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


def test_that_export_tool_generates_a_file(qtbot, opened_main_window_snake_oil):
    gui = opened_main_window_snake_oil

    export_tool = gui.tools["Export data"]

    def handle_export_dialog():
        export_button = wait_for_child(gui, qtbot, QPushButton, name="Export button")

        def close_message_box():
            messagebox = wait_for_child(gui, qtbot, QMessageBox)
            messagebox.close()

        QTimer.singleShot(500, close_message_box)
        qtbot.mouseClick(export_button, Qt.MouseButton.LeftButton)

    QTimer.singleShot(500, handle_export_dialog)
    export_tool.trigger()
    qtbot.waitUntil(lambda: os.path.exists("export.csv"))
