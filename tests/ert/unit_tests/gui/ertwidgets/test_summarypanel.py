from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QScrollArea

from ert.gui.summarypanel import ErtSummary, SummaryPanel


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


def test_that_only_the_overflowing_summary_card_shows_a_scroll_bar(qtbot):
    observations = [{"observation_key": f"OBS_{i}", "count": 1} for i in range(100)]
    with (
        patch.object(ErtSummary, "getForwardModels", return_value=["STEP"]),
        patch.object(ErtSummary, "get_parameters", return_value=(["P (1)"], [], 1)),
        patch.object(ErtSummary, "getObservations", return_value=observations),
    ):
        panel = SummaryPanel(MagicMock())
    qtbot.addWidget(panel)
    panel.resize(900, 300)
    panel.show()

    forward_model, parameters, observations_card = panel.findChildren(QScrollArea)

    assert not forward_model.verticalScrollBar().isVisible()
    assert not parameters.verticalScrollBar().isVisible()
    assert observations_card.verticalScrollBar().isVisible()
