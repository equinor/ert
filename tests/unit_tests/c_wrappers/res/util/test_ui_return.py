import pytest

from ert._c_wrappers.util import UIReturn
from ert._c_wrappers.util.enums import UIReturnStatusEnum


def test_create():
    ui_return = UIReturn(UIReturnStatusEnum.UI_RETURN_OK)
    assert ui_return

    ui_return = UIReturn(UIReturnStatusEnum.UI_RETURN_FAIL)
    assert not ui_return

    assert len(ui_return) == 0


def test_help():
    ui_return = UIReturn(UIReturnStatusEnum.UI_RETURN_OK)
    assert ui_return.help_text() == ""

    ui_return.add_help("Help1")
    assert ui_return.help_text() == "Help1"

    ui_return.add_help("Help2")
    assert ui_return.help_text() == "Help1 Help2"


def test_error_raises_OK():
    ui_return = UIReturn(UIReturnStatusEnum.UI_RETURN_OK)
    with pytest.raises(ValueError):
        ui_return.add_error("Error1")

    with pytest.raises(ValueError):
        ui_return.last_error()

    with pytest.raises(ValueError):
        ui_return.first_error()


def test_add_error():
    ui_return = UIReturn(UIReturnStatusEnum.UI_RETURN_FAIL)
    ui_return.add_error("Error1")
    ui_return.add_error("Error2")
    ui_return.add_error("Error3")
    assert len(ui_return) == 3

    assert ui_return.first_error() == "Error1"
    assert ui_return.last_error() == "Error3"


def test_iget_error():
    ui_return = UIReturn(UIReturnStatusEnum.UI_RETURN_FAIL)
    ui_return.add_error("Error1")
    ui_return.add_error("Error2")
    ui_return.add_error("Error3")

    errorList = []
    for index in range(len(ui_return)):
        errorList.append(ui_return.iget_error(index))
    assert errorList == ["Error1", "Error2", "Error3"]

    with pytest.raises(TypeError):
        ui_return.iget_error("XX")

    ui_return = UIReturn(UIReturnStatusEnum.UI_RETURN_OK)
    errorList = []
    for index in range(len(ui_return)):
        errorList.append(ui_return.iget_error(index))
    assert not errorList
