from unittest.mock import MagicMock

import pytest
from pytestqt.qtbot import QtBot

from ert.config import GenKwConfig
from ert.config.parameter_config import LocalizationType, ParameterConfig
from ert.gui.experiments._update_strategy_summary_widget import (
    UpdateStrategySummaryWidget,
    _summarize_parameters,
)


@pytest.mark.parametrize(
    ("entries", "rows"),
    [
        ([], []),
        (
            [
                ("surface", None, 1),
                ("gen_kw", LocalizationType.ADAPTIVE, 2),
                ("field", LocalizationType.DISTANCE, 1),
                ("gen_kw", LocalizationType.GLOBAL, 3),
                ("gen_kw", None, 1),
            ],
            [
                ("Global", "GenKW", "3"),
                ("Adaptive", "GenKW", "2"),
                ("Distance", "Field", "1"),
                ("Non-updatable", "GenKW", "1"),
                ("Non-updatable", "Surface", "1"),
            ],
        ),
        (
            [("gen_kw", LocalizationType.GLOBAL, 10_000)],
            [("Global", "GenKW", "10,000")],
        ),
        (
            [("custom<type>", None, 1)],
            [("Non-updatable", "custom<type>", "1")],
        ),
    ],
)
def test_that_strategy_summary_counts_configs_by_strategy_and_type(
    entries: list[tuple[str, LocalizationType | None, int]],
    rows: list[tuple[str, str, str]],
) -> None:
    configs = []
    for param_type, strategy, count in entries:
        config = MagicMock(spec=ParameterConfig)
        config.type = param_type
        config.update_strategy = strategy
        config.__len__.return_value = 100
        configs.extend([config] * count)

    assert _summarize_parameters(iter(configs)) == rows


def test_that_strategy_summary_widget_displays_rows_with_headers(
    qtbot: QtBot,
) -> None:
    parameter = GenKwConfig(
        name="parameter",
        distribution={"name": "uniform", "min": 0, "max": 1},
    )
    widget = UpdateStrategySummaryWidget([parameter])
    qtbot.addWidget(widget)
    assert [
        widget.horizontalHeaderItem(column).text()
        for column in range(widget.columnCount())
    ] == ["strategy", "parameter type", "count"]
    assert [
        tuple(widget.item(row, column).text() for column in range(widget.columnCount()))
        for row in range(widget.rowCount())
    ] == [("Global", "GenKW", "1")]


def test_that_replacing_parameters_removes_old_strategy_counts_and_shows_empty_summary(
    qtbot: QtBot,
) -> None:
    parameter = GenKwConfig(
        name="parameter",
        distribution={"name": "uniform", "min": 0, "max": 1},
    )
    widget = UpdateStrategySummaryWidget([parameter])
    qtbot.addWidget(widget)
    widget.show()
    assert widget.rowCount() == 1
    assert widget.item(0, 0).text() == "Global"

    parameter.update_strategy = None
    widget.set_parameters([parameter])
    assert widget.item(0, 0).text() == "Non-updatable"

    widget.set_parameters([])
    assert widget.rowCount() == 0
    widget.set_parameters([parameter])
    assert widget.item(0, 0).text() == "Non-updatable"
