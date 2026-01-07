import re

import pytest
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget

from ert.config import ErrorInfo, WarningInfo
from ert.gui.ertwidgets import Suggestor
from ert.gui.ertwidgets.suggestor.suggestor import _CopyAllButton


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


def test_that_copy_all_button_concatenates_errors_warnings_and_deprecations(qtbot):
    errors = [ErrorInfo("error", filename="script.py", line="5")]
    warnings = [WarningInfo("warning", filename="script.py", line="6")]
    deprecations = [
        WarningInfo("deprecation", filename="script.py", line="7", is_deprecation=True)
    ]
    suggestor = Suggestor(errors, warnings, deprecations, lambda: None)

    all_buttons = suggestor.findChildren(QPushButton)
    copy_all_button = next(
        button for button in all_buttons if isinstance(button, _CopyAllButton)
    )
    copy_all_button.click()

    expected_clipboard = """error
    script.py: Line 5

    warning
    script.py: Line 6

    deprecation
    script.py: Line 7"""

    def remove_whitespaces(s: str) -> str:
        return re.sub(r"\s+", "", s)

    assert remove_whitespaces(QApplication.clipboard().text()) == remove_whitespaces(
        expected_clipboard
    )
