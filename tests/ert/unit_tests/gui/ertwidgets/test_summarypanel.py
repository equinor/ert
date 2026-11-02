from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QLabel

from ert.config import ErtConfig
from ert.gui.summarypanel import SummaryPanel


def test_that_summary_groups_forward_model_steps_by_display_name(qtbot, tmp_path):
    step_file = tmp_path / "copy"
    step_file.write_text("EXECUTABLE echo\n")
    config = ErtConfig.from_file_contents(
        f"""
        NUM_REALIZATIONS 1
        INSTALL_JOB COPY_FILE {step_file}
        FORWARD_MODEL COPY_FILE(<FROM>=a, <TO>=b, NAME=copy_input)
        FORWARD_MODEL COPY_FILE(<FROM>=a, <TO>=b, NAME=copy_input)
        FORWARD_MODEL COPY_FILE(<FROM>=a, <TO>=b, NAME=copy_output_<IENS>)
        FORWARD_MODEL COPY_FILE(<FROM>=a, <TO>=b)
        """
    )
    panel = SummaryPanel(config)
    qtbot.addWidget(panel)
    text = panel.findChildren(QLabel)[0].text()

    assert "Forward model (4 steps)" in text
    assert ">copy_input x2</div>" in text
    assert ">copy_output_&lt;IENS&gt;</div>" in text
    assert ">COPY_FILE</div>" in text


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
