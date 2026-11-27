from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QScrollArea

from ert.gui.ertwidgets.chip import DataSource
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

    forward_model, observations_card = panel.findChildren(QScrollArea)

    qtbot.waitUntil(lambda: observations_card.verticalScrollBar().isVisible())
    assert not forward_model.verticalScrollBar().isVisible()


def test_that_source_chips_follow_the_source_of_parameters_and_observations(qtbot):
    with (
        patch.object(ErtSummary, "getForwardModels", return_value=["STEP"]),
        patch.object(ErtSummary, "get_parameters", return_value=(["P (1)"], [], 1)),
        patch.object(ErtSummary, "getObservations", return_value=[]),
    ):
        panel = SummaryPanel(MagicMock())
    qtbot.addWidget(panel)
    assert panel.parameters_chip.source == DataSource.CONFIG
    assert panel.observations_chip.source == DataSource.CONFIG

    panel.set_parameter_configuration([], DataSource.SELECTED_ENSEMBLE, "exp : ens")
    panel.set_observations(
        [{"observation_key": "OBS", "count": 2}],
        DataSource.SELECTED_ENSEMBLE,
        "exp : ens",
    )

    assert panel.parameters_chip.text() == "From exp : ens"
    assert panel.observations_chip.text() == "From exp : ens"
    assert panel.forward_model_chip.source == DataSource.CONFIG
