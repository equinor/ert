import logging

from ert.gui.tools.plot.customize.customize_plot_dialog import CustomizePlotDialog
from ert.gui.tools.plot.customize.default_customization_view import (
    DefaultCustomizationView,
)


def test_that_first_tab_is_not_logged_when_opening_customize_plot_dialog(qtbot, caplog):
    caplog.set_level(
        logging.INFO,
        logger="ert.gui.tools.plot.customize.customize_plot_dialog",
    )

    plot = CustomizePlotDialog(title="Test Plot", parent=None, key_defs=[])
    plot.addTab("general", "General", DefaultCustomizationView())
    plot.addTab("style", "Style", DefaultCustomizationView())
    qtbot.addWidget(plot)

    plot.show()
    assert "Customization dialog action: General" not in caplog.text

    plot._tabs.setCurrentIndex(1)
    assert "Customization dialog action: Style" in caplog.text

    plot._tabs.setCurrentIndex(0)
    assert "Customization dialog action: General" in caplog.text
    assert len(caplog.records) == 2
