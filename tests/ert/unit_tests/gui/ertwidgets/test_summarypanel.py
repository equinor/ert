from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QLabel

from ert.config import ErtConfig, GenKwConfig
from ert.gui.summarypanel import SummaryPanel


def test_that_parameter_summary_replaces_contents_without_adding_widgets(qtbot):
    panel = SummaryPanel(ErtConfig())
    qtbot.addWidget(panel)
    label_count = len(panel.findChildren(QLabel))
    parameter = GenKwConfig(
        name="prior_parameter", distribution={"name": "uniform", "min": 0, "max": 1}
    )
    panel.set_parameters([parameter])
    assert "Parameters (1)" in panel._parameter_label.text()
    panel.set_parameters(None)
    assert "no ensemble selected" in panel._parameter_label.text()
    panel.set_parameters([])
    assert "Parameters (0)" in panel._parameter_label.text()
    assert len(panel.findChildren(QLabel)) == label_count


@pytest.mark.parametrize(
    ("strings", "expected"),
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
