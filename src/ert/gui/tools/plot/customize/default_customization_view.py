from .customization_view import CustomizationView, WidgetProperty


class DefaultCustomizationView(CustomizationView):
    title = WidgetProperty()
    x_label = WidgetProperty()
    y_label = WidgetProperty()
    legend = WidgetProperty()
    grid = WidgetProperty()
    history = WidgetProperty()
    observations = WidgetProperty()

    def __init__(self):
        # pylint: disable=consider-using-f-string
        CustomizationView.__init__(self)
        label_msg = (
            "Set to empty to use the default %s.\n"
            "It is also possible to use LaTeX. "
            "Enclose expression with $...$ for example: \n"
            "$\\alpha > \\beta$\n"
            "$r^3$\n"
            "$\\frac{1}{x}$\n"
            "$\\sqrt{2}$"
        )

        self.addLineEdit(
            "title",
            "Title",
            f'The title of the plot. {label_msg % "title"}',
            placeholder="Title",
        )
        self.addSpacing()
        self.addLineEdit(
            "x_label",
            "x-label",
            f'The label of the x-axis. {label_msg % "label"}',
            placeholder="x-label",
        )
        self.addLineEdit(
            "y_label",
            "y-label",
            f'The label of the y-axis. {label_msg % "label"}',
            placeholder="y-label",
        )
        self.addSpacing()
        self.addCheckBox("legend", "Legend", "Toggle legend visibility.")
        self.addCheckBox("grid", "Grid", "Toggle grid visibility.")
        self.addCheckBox("history", "History", "Toggle history visibility.")
        self.addCheckBox(
            "observations", "Observations", "Toggle observations visibility."
        )

    def applyCustomization(self, plot_config):
        """
        @type plot_config: ert.gui.plottery.PlotConfig
        """
        plot_config.setTitle(self.title)

        plot_config.setXLabel(self.x_label)
        plot_config.setYLabel(self.y_label)

        plot_config.setLegendEnabled(self.legend)
        plot_config.setGridEnabled(self.grid)
        plot_config.setHistoryEnabled(self.history)
        plot_config.setObservationsEnabled(self.observations)

    def revertCustomization(self, plot_config):
        """
        @type plot_config: ert.gui.plottery.PlotConfig
        """
        if not plot_config.isUnnamed():
            self.title = plot_config.title()
        else:
            self.title = ""

        self.x_label = plot_config.xLabel()
        self.y_label = plot_config.yLabel()

        self.legend = plot_config.isLegendEnabled()
        self.grid = plot_config.isGridEnabled()
        self.history = plot_config.isHistoryEnabled()
        self.observations = plot_config.isObservationsEnabled()
