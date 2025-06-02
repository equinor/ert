from typing import TYPE_CHECKING, cast

from PyQt6.QtWidgets import QHBoxLayout

from .color_chooser import ColorBox
from .customization_view import CustomizationView, WidgetProperty
from .style_chooser import STYLESET_TOGGLE, StyleChooser

if TYPE_CHECKING:
    from ert.gui.tools.plot.plottery import PlotConfig


class StyleCustomizationView(CustomizationView):
    default_style = WidgetProperty()
    history_style = WidgetProperty()
    observs_style = WidgetProperty()
    color_cycle = WidgetProperty()
    observs_color = WidgetProperty()

    def __init__(self) -> None:
        CustomizationView.__init__(self)

        layout = QHBoxLayout()

        self.addRow("", layout)  # type: ignore
        self.addStyleChooser(
            "default_style", "Default", "Line and marker style for default lines."
        )
        self.addStyleChooser(
            "history_style", "History", "Line and marker style for the history line."
        )
        self.addStyleChooser(
            "observs_style",
            "Observation",
            "Line and marker style for the observation line.",
            line_style_set=STYLESET_TOGGLE,
        )

        style = cast(StyleChooser, self["default_style"])
        style.createLabelLayout(layout)

        self.addSpacing(10)

        color_layout = QHBoxLayout()

        self._color_boxes = []
        for name in ["#1", "#2", "#3", "#4", "#5"]:
            color_box = self.createColorBox(name)
            self._color_boxes.append(color_box)
            color_layout.addWidget(color_box)

        self.addRow("Color cycle", color_layout)  # type: ignore
        self.updateProperty(
            "color_cycle",
            StyleCustomizationView.getColorCycle,
            StyleCustomizationView.setColorCycle,
        )

        self._observs_color_box = self.createColorBox("observations_color")
        self.addRow("Observations color", self._observs_color_box)
        self.updateProperty(
            "observs_color",
            StyleCustomizationView.getObservationsColor,
            StyleCustomizationView.setObservationsColor,
        )

    def getObservationsColor(self) -> tuple[str, float]:
        return (
            self._observs_color_box.color.name(),
            self._observs_color_box.color.alphaF(),
        )

    def setObservationsColor(self, color_tuple: tuple[str, float]) -> None:
        self._observs_color_box.color = color_tuple

    @staticmethod
    def createColorBox(name: str) -> ColorBox:
        color_box = ColorBox(size=20)
        color_box.setToolTip(name)
        return color_box

    def getColorCycle(self) -> list[tuple[str, float]]:
        return [
            (color_box.color.name(), color_box.color.alphaF())
            for color_box in self._color_boxes
        ]

    def setColorCycle(self, color_cycle: list[tuple[str, float]]) -> None:
        for index, color_tuple in enumerate(color_cycle):
            if 0 <= index < len(self._color_boxes):
                color_box = self._color_boxes[index]
                color_box.color = color_tuple

    def applyCustomization(self, plot_config: "PlotConfig") -> None:
        plot_config.setDefaultStyle(self.default_style)
        plot_config.setHistoryStyle(self.history_style)
        plot_config.setObservationsStyle(self.observs_style)
        plot_config.setObservationsColor(self.observs_color)
        plot_config.setLineColorCycle(self.color_cycle)

    def revertCustomization(self, plot_config: "PlotConfig") -> None:
        self.default_style = plot_config.defaultStyle()
        self.history_style = plot_config.historyStyle()
        self.observs_style = plot_config.observationsStyle()
        self.observs_color = plot_config.observationsColor()
        self.color_cycle = plot_config.lineColorCycle()
