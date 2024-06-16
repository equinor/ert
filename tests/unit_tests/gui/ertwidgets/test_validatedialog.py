from ert.gui.ertwidgets.validateddialog import ValidatedDialog


def test_toggling_valid_will_toggle_ok_button_enabled(qtbot):
    dialog = ValidatedDialog()
    qtbot.addWidget(dialog)

    assert not dialog.ok_button.isEnabled()

    dialog.valid()
    assert dialog.ok_button.isEnabled()

    dialog.notValid("Not valid")
    assert not dialog.ok_button.isEnabled()
    assert dialog.param_name.toolTip() == "Not valid"


def test_validate_name(qtbot):
    dialog = ValidatedDialog(unique_names=["name1", "name2"])
    qtbot.addWidget(dialog)

    dialog.validateName("John Snow")
    assert dialog.param_name.toolTip() == "No spaces allowed!"

    dialog.validateName("name1")
    assert dialog.param_name.toolTip() == "Name must be unique!"

    dialog.validateName("unique_name")
    assert not dialog.param_name.toolTip()
    assert dialog.ok_button.isEnabled()
