import pytest
from ert_gui.ertwidgets.summarypanel import _runlength_encode_list


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
def test_runlength_encode_list(strings, expected):
    assert _runlength_encode_list(strings) == expected
