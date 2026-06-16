import pytest
from pytestqt.qtbot import QtBot

from ert.gui.tools.plot.data_type_keys_widget import DataTypeKeysWidget
from ert.gui.tools.plot.plot_api import PlotApiKeyDefinition


def create_key_def(key: str) -> PlotApiKeyDefinition:
    return PlotApiKeyDefinition(
        key=key,
        index_type=None,
        observations=False,
        dimensionality=1,
        metadata={"data_origin": "summary"},
    )


@pytest.fixture
def key_defs() -> list[PlotApiKeyDefinition]:
    return [
        create_key_def("A"),
        create_key_def("B"),
    ]


def test_that_filtering_out_current_key_does_not_emit_data_type_key_selected(
    qtbot: QtBot,
    key_defs: list[PlotApiKeyDefinition],
) -> None:
    """Filtering out the current key should clear selection without emitting."""
    widget = DataTypeKeysWidget(key_defs)
    qtbot.addWidget(widget)

    widget.selectDefault()
    assert widget.getSelectedItem() == key_defs[0]

    with qtbot.assertNotEmitted(widget.dataTypeKeySelected):
        widget.setSearchString("B")

    assert widget.getSelectedItem() is None


def test_that_filtering_keeps_current_key_when_still_visible(
    qtbot: QtBot,
    key_defs: list[PlotApiKeyDefinition],
) -> None:
    """Filtering should keep the current key selected when it remains visible."""
    widget = DataTypeKeysWidget(key_defs)
    qtbot.addWidget(widget)

    widget.selectDefault()
    assert widget.getSelectedItem() == key_defs[0]

    with qtbot.assertNotEmitted(widget.dataTypeKeySelected):
        widget.setSearchString("A")

    assert widget.data_type_keys_widget.currentIndex().isValid()
    assert widget.getSelectedItem() == key_defs[0]
