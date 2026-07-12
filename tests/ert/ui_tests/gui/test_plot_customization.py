import logging

from ert.gui.plotting.customization_dialog.customize_plot_dialog import (
    CustomizePlotDialog,
)
from ert.gui.plotting.customization_dialog.statistics_customization_view import (
    StatisticsCustomizationView,
)
from ert.gui.plotting.customization_dialog.style_customization_view import (
    StyleCustomizationView,
)


def test_that_first_tab_is_not_logged_when_opening_customize_plot_dialog(qtbot, caplog):
    caplog.set_level(
        logging.INFO,
        logger="ert.gui.plotting.customization_dialog.customize_plot_dialog",
    )

    plot = CustomizePlotDialog(title="Test Plot", parent=None, key_defs=[])
    plot.add_tab("style", "Style", StyleCustomizationView())
    plot.add_tab("statistics", "Statistics", StatisticsCustomizationView())
    qtbot.addWidget(plot)

    plot.show()
    assert "Customization dialog action: Style" not in caplog.text

    plot._tabs.setCurrentIndex(1)
    assert "Customization dialog action: Statistics" in caplog.text

    plot._tabs.setCurrentIndex(0)
    assert "Customization dialog action: Style" in caplog.text
    assert len(caplog.records) == 2
