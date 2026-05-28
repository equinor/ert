import logging

import pytest

from ert.gui.tools.plot.customize.customize_plot_dialog import CustomizePlotDialog
from ert.gui.tools.plot.customize.default_customization_view import (
    DefaultCustomizationView,
)
from ert.gui.tools.plot.customize.style_customization_view import StyleCustomizationView
from ert.gui.tools.plot.plot_api import PlotApiKeyDefinition


def test_that_first_tab_is_not_logged_when_opening_customize_plot_dialog(qtbot, caplog):
    caplog.set_level(
        logging.INFO,
        logger="ert.gui.tools.plot.customize.customize_plot_dialog",
    )

    plot = CustomizePlotDialog(title="Test Plot", parent=None, key_defs=[])
    plot.add_tab("general", "General", DefaultCustomizationView())
    plot.add_tab("style", "Style", DefaultCustomizationView())
    qtbot.addWidget(plot)

    plot.show()
    assert "Customization dialog action: General" not in caplog.text

    plot._tabs.setCurrentIndex(1)
    assert "Customization dialog action: Style" in caplog.text

    plot._tabs.setCurrentIndex(0)
    assert "Customization dialog action: General" in caplog.text
    assert len(caplog.records) == 2


@pytest.mark.parametrize(
    (
        "key_def",
        "history_visible",
        "observations_visible",
        "observations_style_visible",
    ),
    [
        (
            PlotApiKeyDefinition(
                key="MY_PARAM",
                index_type=None,
                observations=False,
                dimensionality=1,
                metadata={"data_origin": "gen_kw"},
            ),
            False,
            False,
            False,
        ),
        (
            PlotApiKeyDefinition(
                key="MY_FIELD",
                index_type=None,
                observations=False,
                dimensionality=3,
                metadata={"data_origin": "field"},
            ),
            False,
            True,
            False,
        ),
        (
            PlotApiKeyDefinition(
                key="MY_SURFACE",
                index_type=None,
                observations=False,
                dimensionality=3,
                metadata={"data_origin": "surface"},
            ),
            False,
            False,
            False,
        ),
        (
            PlotApiKeyDefinition(
                key="WWCT:OP_1",
                index_type="VALUE",
                observations=True,
                dimensionality=2,
                metadata={"data_origin": "summary"},
                response=object(),  # type: ignore[arg-type]
            ),
            True,
            True,
            True,
        ),
    ],
)
def test_observation_and_history_settings_visibility(
    qtbot,
    key_def,
    history_visible,
    observations_visible,
    observations_style_visible,
):
    general_view = DefaultCustomizationView()
    style_view = StyleCustomizationView()
    qtbot.addWidget(general_view)
    qtbot.addWidget(style_view)

    general_view.set_key_definition(key_def)
    style_view.set_key_definition(key_def)

    assert general_view["history"].isVisibleTo(general_view) is history_visible
    assert (
        general_view["observations"].isVisibleTo(general_view) is observations_visible
    )
    assert style_view["history_style"].isVisibleTo(style_view) is history_visible
    assert (
        style_view["observations_style"].isVisibleTo(style_view)
        is observations_style_visible
    )
    assert (
        style_view["observations_color"].isVisibleTo(style_view)
        is observations_style_visible
    )
