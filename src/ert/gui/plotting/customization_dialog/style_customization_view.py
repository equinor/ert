from typing import TYPE_CHECKING, cast, override

from PyQt6.QtWidgets import QHBoxLayout

from .customization_view import CustomizationView, WidgetProperty
from .style_chooser import STYLESET_TOGGLE, StyleChooser

if TYPE_CHECKING:
    from ert.gui.plotting.utils import PlotConfig


class StyleCustomizationView(CustomizationView):
    default_style = WidgetProperty()
    history_style = WidgetProperty()
    observations_style = WidgetProperty()

    def __init__(self) -> None:
        CustomizationView.__init__(self)

        layout = QHBoxLayout()

        self.add_row("", layout)  # type: ignore
        self.add_style_chooser(
            "default_style", "Default", "Line and marker style for default lines."
        )
        self.add_style_chooser(
            "history_style", "History", "Line and marker style for the history line."
        )
        self.add_style_chooser(
            "observations_style",
            "Observation",
            "Line and marker style for the observation line.",
            line_style_set=STYLESET_TOGGLE,
        )

        style = cast(StyleChooser, self["default_style"])
        style.create_label_layout(layout)

        self.add_spacing(10)

    @override
    def apply_customization(self, plot_config: "PlotConfig") -> None:
        plot_config.set_default_style(self.default_style)
        plot_config.set_history_style(self.history_style)
        plot_config.set_observations_style(self.observations_style)

    @override
    def revert_customization(self, plot_config: "PlotConfig") -> None:
        self.default_style = plot_config.default_style()
        self.history_style = plot_config.history_style()
        self.observations_style = plot_config.observations_style()
