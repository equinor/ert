from typing import TYPE_CHECKING, cast, override

from PyQt6.QtWidgets import QHBoxLayout

from .color_chooser import ColorBox
from .customization_view import CustomizationView, WidgetProperty
from .style_chooser import STYLESET_TOGGLE, StyleChooser

if TYPE_CHECKING:
    from ert.gui.plotting.utils import PlotConfig


class StyleCustomizationView(CustomizationView):
    default_style = WidgetProperty()
    history_style = WidgetProperty()
    observations_style = WidgetProperty()
    color_cycle = WidgetProperty()
    observations_color = WidgetProperty()

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

        color_layout = QHBoxLayout()

        self._color_boxes = []
        for name in ["#1", "#2", "#3", "#4", "#5"]:
            color_box = self.create_color_box(name)
            self._color_boxes.append(color_box)
            color_layout.addWidget(color_box)

        self.add_row("Color cycle", color_layout)  # type: ignore
        self.update_property(
            "color_cycle",
            StyleCustomizationView.get_color_cycle,
            StyleCustomizationView.set_color_cycle,
        )

        self._observations_color_box = self.create_color_box("observations_color")
        self.add_row("Observations color", self._observations_color_box)
        self.update_property(
            "observations_color",
            StyleCustomizationView.get_observations_color,
            StyleCustomizationView.set_observations_color,
        )

    def get_observations_color(self) -> tuple[str, float]:
        return (
            self._observations_color_box.color.name(),
            self._observations_color_box.color.alphaF(),
        )

    def set_observations_color(self, color_tuple: tuple[str, float]) -> None:
        self._observations_color_box.color = color_tuple

    @staticmethod
    def create_color_box(name: str) -> ColorBox:
        color_box = ColorBox(size=20)
        color_box.setToolTip(name)
        return color_box

    def get_color_cycle(self) -> list[tuple[str, float]]:
        return [
            (color_box.color.name(), color_box.color.alphaF())
            for color_box in self._color_boxes
        ]

    def set_color_cycle(self, color_cycle: list[tuple[str, float]]) -> None:
        for index, color_tuple in enumerate(color_cycle):
            if 0 <= index < len(self._color_boxes):
                color_box = self._color_boxes[index]
                color_box.color = color_tuple

    @override
    def apply_customization(self, plot_config: "PlotConfig") -> None:
        plot_config.set_default_style(self.default_style)
        plot_config.set_history_style(self.history_style)
        plot_config.set_observations_style(self.observations_style)
        plot_config.set_observations_color(self.observations_color)
        plot_config.set_line_color_cycle(self.color_cycle)

    @override
    def revert_customization(self, plot_config: "PlotConfig") -> None:
        self.default_style = plot_config.default_style()
        self.history_style = plot_config.history_style()
        self.observations_style = plot_config.observations_style()
        self.observations_color = plot_config.observations_color()
        self.color_cycle = plot_config.line_color_cycle()
