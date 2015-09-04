class PlotTools(object):
    @staticmethod
    def showGrid(axes, plot_context):
        config = plot_context.plotConfig()
        if config.isGridEnabled():
            axes.grid()


    @staticmethod
    def showLegend(axes, plot_context):
        config = plot_context.plotConfig()
        if config.isLegendEnabled():
            axes.legend(config.legendItems(), config.legendLabels())

