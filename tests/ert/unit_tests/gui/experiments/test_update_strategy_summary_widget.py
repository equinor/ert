from unittest.mock import MagicMock

import pytest
from pytestqt.qtbot import QtBot

from ert.config import GenKwConfig
from ert.config.parameter_config import LocalizationType, ParameterConfig
from ert.gui.ertwidgets.models.parameter_configuration_state_model import (
    ParameterConfigurationStateModel,
)
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
                ("Adaptive", "GenKW", "2"),
                ("Distance", "Field", "1"),
                ("Global", "GenKW", "3"),
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
def test_that_strategy_summary_orders_counts_lexicographically_by_strategy_and_type(
    entries: list[tuple[str, LocalizationType | None, int]],
    rows: list[tuple[str, str, str]],
):
    configs = []
    for param_type, strategy, count in entries:
        config = MagicMock(spec=ParameterConfig)
        config.type = param_type
        config.update_strategy = strategy
        config.__len__.return_value = 100
        configs.extend([config] * count)

    assert _summarize_parameters(iter(configs)) == rows


def _parameter(
    update_strategy: LocalizationType | None = LocalizationType.GLOBAL,
) -> GenKwConfig:
    return GenKwConfig(
        name="parameter",
        distribution={"name": "uniform", "min": 0, "max": 1},
        update_strategy=update_strategy,
    )


def _rows(widget: UpdateStrategySummaryWidget) -> list[tuple[str, ...]]:
    return [
        tuple(widget.item(row, column).text() for column in range(widget.columnCount()))
        for row in range(widget.rowCount())
    ]


def test_that_strategy_summary_widget_displays_rows_with_headers(qtbot: QtBot):
    widget = UpdateStrategySummaryWidget(
        ParameterConfigurationStateModel([_parameter()])
    )
    qtbot.addWidget(widget)
    assert [
        widget.horizontalHeaderItem(column).text()
        for column in range(widget.columnCount())
    ] == ["strategy", "parameter type", "count"]
    assert _rows(widget) == [("Global", "GenKW", "1")]


def test_that_strategy_summary_widget_follows_parameter_state_changes(
    qtbot: QtBot,
):
    state = ParameterConfigurationStateModel([_parameter()])
    widget = UpdateStrategySummaryWidget(state)
    qtbot.addWidget(widget)

    state.apply_update_strategies({"GEN_KW": LocalizationType.ADAPTIVE})
    assert _rows(widget) == [("Adaptive", "GenKW", "1")]

    state.select_prior([_parameter(update_strategy=None)])
    assert _rows(widget) == [("Non-updatable", "GenKW", "1")]

    state.select_prior([])
    assert _rows(widget) == []

    state.deselect_prior()
    assert _rows(widget) == [("Adaptive", "GenKW", "1")]
