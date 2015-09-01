from ert.enkf.export import GenKwCollector
from ert_gui.plottery import PlotTools, ProbabilityPlot, PlotContext


class GenKwPlot(object):
    @staticmethod
    def gaussianKDE(plot_context):
        """
        @type plot_context: PlotContext
        """
        ert = plot_context.ert()
        key = plot_context.key()
        config = plot_context.plotConfig()
        axes = plot_context.figure().add_subplot(111)
        """:type: matplotlib.axes.Axes """

        if config.xLabel() is None:
            config.setXLabel("Value")

        if config.yLabel() is None:
            config.setYLabel("Density")

        axes.set_title(config.title())

        if key.startswith("LOG10_"):
            key = key[6:]
            axes.set_xscale("log")

        case_list = plot_context.cases()
        for case in case_list:
            data = GenKwCollector.loadAllGenKwData(ert, case, [key])

            if not data.empty:
                ProbabilityPlot.plotGaussianKDE(axes, config, data[key].values, case)

                config.nextColor()

        PlotTools.showLegend(axes, plot_context)
        PlotTools.showGrid(axes, plot_context)


    @staticmethod
    def histogram(plot_context):
        """
        @type plot_context: PlotContext
        """
        ert = plot_context.ert()
        key = plot_context.key()
        config = plot_context.plotConfig()
        case_list = plot_context.cases()
        case_count = len(case_list)


        if config.xLabel() is None:
            config.setXLabel("Value")

        if config.yLabel() is None:
            config.setYLabel("Count")

        use_log_scale = False
        if key.startswith("LOG10_"):
            key = key[6:]
            use_log_scale = True

        data = {}
        minimum = None
        maximum = None
        for case in case_list:
            data[case] = GenKwCollector.loadAllGenKwData(ert, case, [key])
            
            if minimum is None:
                minimum = data[case][key].min()
            else:
                minimum = min(minimum, data[case][key].min())
            
            if maximum is None:
                maximum = data[case][key].max()
            else:
                maximum = max(maximum, data[case][key].max())

        axes = {}
        """:type: dict of (Str, matplotlib.axes.Axes) """
        for index, case in enumerate(case_list):
            axes[case] = plot_context.figure().add_subplot(case_count, 1, index + 1)


            axes[case].set_title("%s (%s)" % (config.title(), case))

            if use_log_scale:
                axes[case].set_xscale("log")

            if not data[case].empty:
                ProbabilityPlot.plotHistogram(axes[case], config, data[case][key], case, use_log_scale, minimum, maximum)

                config.nextColor()
                PlotTools.showGrid(axes[case], plot_context)

        max_count = max([subplot.get_ylim()[1] for subplot in axes.values()])

        for subplot in axes.values():
            subplot.set_ylim(0, max_count)
