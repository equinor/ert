import logging

from ert.gui.plotting.customization_dialog.customization_view import CustomizationView
from ert.gui.plotting.customization_dialog.customize_plot_dialog import (
    CustomizePlotDialog,
)


class _DummyCustomizationView(CustomizationView):
    def apply_customization(self, plot_config):
        pass

    def revert_customization(self, plot_config):
        pass


def test_that_first_tab_is_not_logged_when_opening_customize_plot_dialog(qtbot, caplog):
    caplog.set_level(
        logging.INFO,
        logger="ert.gui.plotting.customization_dialog.customize_plot_dialog",
    )

    plot = CustomizePlotDialog(title="Test Plot", parent=None, key_defs=[])
    plot.add_tab("first", "First", _DummyCustomizationView())
    plot.add_tab("second", "Second", _DummyCustomizationView())
    qtbot.addWidget(plot)

    plot.show()
    assert "Customization dialog action: First" not in caplog.text

    plot._tabs.setCurrentIndex(1)
    assert "Customization dialog action: Second" in caplog.text

    plot._tabs.setCurrentIndex(0)
    assert "Customization dialog action: First" in caplog.text
    assert len(caplog.records) == 2
