from typing import TYPE_CHECKING

from typing_extensions import override

from ert.gui.utils import is_everest_application

from .customization_view import CustomizationView, WidgetProperty

if TYPE_CHECKING:
    from ert.gui.plotting.utils import PlotConfig


def _label_msg(label: str) -> str:
    return (
        f"Set to empty to use the default {label}.\n"
        "It is also possible to use LaTeX. "
        "Enclose expression with $...$ for example: \n"
        "$\\alpha > \\beta$\n"
        "$r^3$\n"
        "$\\frac{1}{x}$\n"
        "$\\sqrt{2}$"
    )


class DefaultCustomizationView(CustomizationView):
    title = WidgetProperty()
    x_label = WidgetProperty()
    y_label = WidgetProperty()
    legend = WidgetProperty()
    grid = WidgetProperty()
    history = WidgetProperty()
    observations = WidgetProperty()

    def __init__(self) -> None:
        CustomizationView.__init__(self)

        self.add_line_edit(
            "title",
            "Title",
            f"The title of the plot. {_label_msg('title')}",
            placeholder="Title",
        )
        self.add_spacing()
        self.add_line_edit(
            "x_label",
            "x-label",
            f"The label of the x-axis. {_label_msg('label')}",
            placeholder="x-label",
        )
        self.add_line_edit(
            "y_label",
            "y-label",
            f"The label of the y-axis. {_label_msg('label')}",
            placeholder="y-label",
        )
        self.add_spacing()
        self.add_check_box("legend", "Legend", "Toggle legend visibility.")
        self.add_check_box("grid", "Grid", "Toggle grid visibility.")

        self.is_everest = is_everest_application()
        if not self.is_everest:
            self.add_check_box("history", "History", "Toggle history visibility.")
            self.add_check_box(
                "observations", "Observations", "Toggle observations visibility."
            )

    @override
    def apply_customization(self, plot_config: "PlotConfig") -> None:
        plot_config.set_title(self.title)

        plot_config.set_x_label(self.x_label)
        plot_config.set_y_label(self.y_label)

        plot_config.set_legend_enabled(self.legend)
        plot_config.set_grid_enabled(self.grid)
        if not self.is_everest:
            plot_config.set_history_enabled(self.history)
            plot_config.set_observations_enabled(self.observations)

    @override
    def revert_customization(self, plot_config: "PlotConfig") -> None:
        if not plot_config.is_unnamed():
            self.title = plot_config.title()
        else:
            self.title = ""

        self.x_label = plot_config.x_label()
        self.y_label = plot_config.y_label()

        self.legend = plot_config.is_legend_enabled()
        self.grid = plot_config.is_grid_enabled()
        if not self.is_everest:
            self.history = plot_config.is_history_enabled()
            self.observations = plot_config.is_observations_enabled()
