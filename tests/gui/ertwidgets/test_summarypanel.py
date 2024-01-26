from unittest.mock import MagicMock

import pytest

from ert.gui.ertwidgets.summarypanel import SummaryPanel


@pytest.mark.parametrize(
    "strings, expected",
    [
        ([], []),
        ([""], [("", 1)]),
        (["foo"], [("foo", 1)]),
        (["foo", "bar"], [("foo", 1), ("bar", 1)]),
        (["foo", "foo"], [("foo", 2)]),
        (["foo", "foo", "foo"], [("foo", 3)]),
        (["foo", "bar", "foo"], [("foo", 1), ("bar", 1), ("foo", 1)]),
    ],
)
def test_runlength_encode_list(qtbot, strings, expected):
    panel = SummaryPanel(MagicMock())
    assert panel._runlength_encode_list(strings) == expected
