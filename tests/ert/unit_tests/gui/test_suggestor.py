import pytest
from PyQt6.QtWidgets import QWidget

from ert.config import ErrorInfo
from ert.gui.ertwidgets import Suggestor


@pytest.mark.parametrize(
    ("errors", "expected_num"),
    [
        ([ErrorInfo("msg_1")], 1),
        ([ErrorInfo("msg_1"), ErrorInfo("msg_2")], 2),
        ([ErrorInfo("msg_1"), ErrorInfo("msg_1"), ErrorInfo("msg_2")], 2),
        ([ErrorInfo("msg_1"), ErrorInfo("msg_2"), ErrorInfo("msg_3")], 3),
    ],
)
def test_suggestor_combines_errors_with_the_same_message(qtbot, errors, expected_num):
    suggestor = Suggestor(errors, [], [], lambda: None)
    msgs = suggestor.findChild(QWidget, name="suggestor_messages")
    assert msgs is not None
    msg_layout = msgs.layout()
    assert msg_layout is not None
    assert msg_layout.count() == expected_num
