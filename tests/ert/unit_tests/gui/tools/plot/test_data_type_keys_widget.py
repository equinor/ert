import pytest
from pytestqt.qtbot import QtBot

from ert.gui.plotting.plot_api import PlotApiKeyDefinition
from ert.gui.plotting.widgets import DataTypeKeysWidget


def create_key_def(key: str, data_origin: str = "summary") -> PlotApiKeyDefinition:
    return PlotApiKeyDefinition(
        key=key,
        index_type=None,
        observations=False,
        dimensionality=1,
        metadata={"data_origin": data_origin},
    )


@pytest.fixture
def key_defs() -> list[PlotApiKeyDefinition]:
    return [
        create_key_def("A"),
        create_key_def("B", data_origin="gen_data"),
    ]


def test_that_filtering_out_current_key_does_not_emit_data_type_key_selected(
    qtbot: QtBot,
    key_defs: list[PlotApiKeyDefinition],
) -> None:
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
    widget = DataTypeKeysWidget(key_defs)
    qtbot.addWidget(widget)

    widget.selectDefault()
    assert widget.getSelectedItem() == key_defs[0]

    with qtbot.assertNotEmitted(widget.dataTypeKeySelected):
        widget.setSearchString("A")

    assert widget.data_type_keys_widget.currentIndex().isValid()
    assert widget.getSelectedItem() == key_defs[0]


def test_that_metadata_filtering_current_key_does_not_emit_data_type_key_selected(
    qtbot: QtBot,
    key_defs: list[PlotApiKeyDefinition],
) -> None:
    widget = DataTypeKeysWidget(key_defs)
    qtbot.addWidget(widget)

    # Select a non-default key before applying the metadata filter.
    selected_index = widget.filter_model.index(1, 0)
    widget.data_type_keys_widget.setCurrentIndex(selected_index)
    assert widget.getSelectedItem() == key_defs[1]

    # Hide the metadata group containing the current key.
    with qtbot.assertNotEmitted(widget.dataTypeKeySelected):
        widget.onItemChanged({"gen_data": False})

    assert widget.getSelectedItem() is None
