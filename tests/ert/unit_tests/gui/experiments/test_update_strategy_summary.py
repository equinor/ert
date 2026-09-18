from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import Qt
from pytestqt.qtbot import QtBot

from ert.config import GenKwConfig
from ert.config.parameter_config import LocalizationType, ParameterConfig
from ert.gui.experiments._update_strategy_summary import UpdateStrategySummary


@pytest.mark.parametrize(
    ("entries", "summary"),
    [
        (
            [
                ("surface", None, 1),
                ("gen_kw", LocalizationType.ADAPTIVE, 2),
                ("field", LocalizationType.DISTANCE, 1),
                ("gen_kw", LocalizationType.GLOBAL, 3),
                ("gen_kw", None, 1),
            ],
            (
                "Global: 3 GenKW; Adaptive: 2 GenKW; Distance: 1 Field; "
                "Non-updatable: 1 GenKW, 1 Surface"
            ),
        ),
        (
            [("gen_kw", LocalizationType.GLOBAL, 10_000)],
            "Global: 10,000 GenKW",
        ),
        (
            [("custom<type>", None, 1)],
            "Non-updatable: 1 custom<type>",
        ),
    ],
)
def test_that_strategy_summary_counts_configs_without_listing_parameter_names(
    qtbot: QtBot, entries, summary
) -> None:
    configs = []
    for param_type, strategy, count in entries:
        config = MagicMock(spec=ParameterConfig)
        config.type = param_type
        config.update_strategy = strategy
        config.__len__.return_value = 100
        configs.extend([config] * count)

    widget = UpdateStrategySummary(iter(configs))
    qtbot.addWidget(widget)
    assert widget.text() == summary
    assert widget.textFormat() == Qt.TextFormat.PlainText


def test_that_replacing_parameters_removes_old_strategy_counts_and_shows_empty_summary(
    qtbot: QtBot,
) -> None:
    parameter = GenKwConfig(
        name="parameter",
        distribution={"name": "uniform", "min": 0, "max": 1},
    )
    widget = UpdateStrategySummary([parameter])
    qtbot.addWidget(widget)
    widget.show()
    assert widget.text() == "Global: 1 GenKW"

    parameter.update_strategy = None
    widget.set_parameters([parameter])
    assert widget.text() == "Non-updatable: 1 GenKW"

    widget.set_parameters([])
    assert widget.isVisible()
    assert widget.text() == "None"
    widget.set_parameters([parameter])
    assert not widget.isHidden()
    assert widget.text() == "Non-updatable: 1 GenKW"
