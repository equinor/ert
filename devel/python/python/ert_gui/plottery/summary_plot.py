from pandas import DataFrame
from ert.ecl import EclSum
from ert.enkf.export import SummaryCollector, SummaryObservationCollector
from ert_gui.plottery import PlotConfig, PlotContext, EnsemblePlot, ObservationPlot


class SummaryPlot(object):
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


    @staticmethod
    def finalizePlot(plot_context, axes):
        SummaryPlot.plotObservations(axes, plot_context)
        SummaryPlot.plotRefcase(axes, plot_context)
        SummaryPlot.showLegend(axes, plot_context)
        SummaryPlot.showGrid(axes, plot_context)
        plot_context.figure().autofmt_xdate()


    @staticmethod
    def summaryEnsemblePlot(plot_context):
        """
        @type plot_context: PlotContext
        """
        ert = plot_context.ert()
        key = plot_context.key()
        config = plot_context.plotConfig()
        axes = plot_context.figure().add_subplot(111)
        """:type: matplotlib.axes.Axes """

        SummaryPlot.setupLabels(plot_context)

        axes.set_title(config.title())

        case_list = plot_context.cases()
        for case in case_list:
            data = SummaryCollector.loadAllSummaryData(ert, case, [key])
            if not data.empty:
                data = data.reset_index()
                data = data.pivot(index="Date", columns="Realization", values=key)

                EnsemblePlot.plotLines(axes, config, data, case)
                config.nextColor()

        SummaryPlot.finalizePlot(plot_context, axes)


    @staticmethod
    def summaryOverviewPlot(plot_context):
        """
        @type plot_context: PlotContext
        """
        ert = plot_context.ert()
        key = plot_context.key()
        config = plot_context.plotConfig()
        axes = plot_context.figure().add_subplot(111)
        """:type: matplotlib.axes.Axes """

        SummaryPlot.setupLabels(plot_context)

        axes.set_title(config.title())

        case_list = plot_context.cases()
        for case in case_list:
            data = SummaryCollector.loadAllSummaryData(ert, case, [key])
            if not data.empty:
                data = data.reset_index()
                data = data.pivot(index="Date", columns="Realization", values=key)

                min_max_data = DataFrame()
                min_max_data["Minimum"] = data.min(axis=1)
                min_max_data["Maximum"] = data.max(axis=1)

                EnsemblePlot.plotArea(axes, config, min_max_data, case)
                config.nextColor()

        SummaryPlot.finalizePlot(plot_context, axes)


    @staticmethod
    def summaryStatisticsPlot(plot_context):
        """
        @type plot_context: PlotContext
        """
        ert = plot_context.ert()
        key = plot_context.key()
        config = plot_context.plotConfig()
        axes = plot_context.figure().add_subplot(111)
        """:type: matplotlib.axes.Axes """

        SummaryPlot.setupLabels(plot_context)

        axes.set_title(config.title())

        case_list = plot_context.cases()
        for case in case_list:
            data = SummaryCollector.loadAllSummaryData(ert, case, [key])
            if not data.empty:
                data = data.reset_index()
                data = data.pivot(index="Date", columns="Realization", values=key)

                statistics_data = DataFrame()

                statistics_data["Minimum"] = data.min(axis=1)
                statistics_data["Maximum"] = data.max(axis=1)
                statistics_data["Mean"] = data.mean(axis=1)
                statistics_data["p10"] = data.quantile(0.1, axis=1)
                statistics_data["p33"] = data.quantile(0.33, axis=1)
                statistics_data["p50"] = data.quantile(0.50, axis=1)
                statistics_data["p67"] = data.quantile(0.67, axis=1)
                statistics_data["p90"] = data.quantile(0.90, axis=1)

                EnsemblePlot.plotPercentiles(axes, config, statistics_data, case)
                config.nextColor()

        SummaryPlot.finalizePlot(plot_context, axes)


    @staticmethod
    def plotRefcase(axes, plot_context):
        ert = plot_context.ert()
        key = plot_context.key()
        config = plot_context.plotConfig()

        if config.isRefcaseEnabled() and ert.eclConfig().hasRefcase():
            refcase_data = SummaryPlot.getRefcaseData(ert, key)

            if not refcase_data.empty:
                EnsemblePlot.plotRefcase(axes, config, refcase_data)



    @staticmethod
    def plotObservations(axes, plot_context):
        ert = plot_context.ert()
        key = plot_context.key()
        config = plot_context.plotConfig()
        case_list = plot_context.cases()

        if config.isObservationsEnabled() and ert.getKeyManager().isKeyWithObservations(key):
            if len(case_list) > 0:
                observation_data = SummaryObservationCollector.loadObservationData(ert, case_list[0], [key])

                if not observation_data.empty:
                    ObservationPlot.plot(axes, config, observation_data, value_column=key)


    @staticmethod
    def setupLabels(plot_context):
        ert = plot_context.ert()
        key = plot_context.key()
        config = plot_context.plotConfig()

        if config.xLabel() is None:
            config.setXLabel("Date")

        if config.yLabel() is None:
            config.setYLabel("Value")

            if ert.eclConfig().hasRefcase():
                unit = ert.eclConfig().getRefcase().unit(key)
                if unit != "":
                    config.setYLabel(unit)


    @staticmethod
    def getRefcaseData(ert, key):
        refcase = ert.eclConfig().getRefcase()
        vector = refcase.get_vector(key, report_only=False)

        rows = []
        for index in range(1, len(vector)):
            node = vector[index]
            row = {
                "Date": EclSum.cNamespace().get_report_time(refcase, node.report_step).datetime(),
                key: node.value
            }
            rows.append(row)

        data = DataFrame(rows)
        data = data.set_index("Date")

        return data
