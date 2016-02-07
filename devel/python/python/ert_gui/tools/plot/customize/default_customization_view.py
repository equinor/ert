from ert_gui.tools.plot.customize import CustomizationView, WidgetProperty


class DefaultCustomizationView(CustomizationView):

    title = WidgetProperty()
    x_label = WidgetProperty()
    y_label = WidgetProperty()
    legend = WidgetProperty()
    grid = WidgetProperty()
    refcase = WidgetProperty()
    observations = WidgetProperty()
    distribution_lines = WidgetProperty()

    def __init__(self):
        CustomizationView.__init__(self)

        self.addLineEdit("title", "Title", "The title of the plot. Set to empty to use the default title.", placeholder="Title")
        self.addSpacing()
        self.addLineEdit("x_label", "X Label", "The label of the X axis. Set to empty to use the default label.", placeholder="X Label")
        self.addLineEdit("y_label", "Y Label", "The label of the Y axis. Set to empty to use the default label.", placeholder="Y Label")
        self.addSpacing()
        self.addCheckBox("legend", "Legend", "Toggle Legend visibility.")
        self.addCheckBox("grid", "Grid", "Toggle Grid visibility.")
        self.addCheckBox("refcase", "Refcase", "Toggle Refcase visibility.")
        self.addCheckBox("observations", "Observations", "Toggle Observations visibility.")
        self.addCheckBox("distribution_lines", "Connection Lines", "Toggle distribution connection lines visibility.")

    def applyCustomization(self, plot_config):
        """
        @type plot_config: ert_gui.plottery.PlotConfig
        """
        plot_config.setTitle(self.title)

        plot_config.setXLabel(self.x_label)
        plot_config.setYLabel(self.y_label)

        plot_config.setLegendEnabled(self.legend)
        plot_config.setGridEnabled(self.grid)
        plot_config.setRefcaseEnabled(self.refcase)
        plot_config.setObservationsEnabled(self.observations)
        plot_config.setDistributionLineEnabled(self.distribution_lines)

    def revertCustomization(self, plot_config):
        """
        @type plot_config: ert_gui.plottery.PlotConfig
        """
        if not plot_config.isUnnamed():
            self.title = plot_config.title()
        else:
            self.title = ""

        self.x_label = plot_config.xLabel()
        self.y_label = plot_config.yLabel()

        self.legend = plot_config.isLegendEnabled()
        self.grid = plot_config.isGridEnabled()
        self.refcase = plot_config.isRefcaseEnabled()
        self.observations = plot_config.isObservationsEnabled()
        self.distribution_lines = plot_config.isDistributionLineEnabled()
