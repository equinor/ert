from pandas import DataFrame
from ert.ecl import EclSum
from ert.enkf.export import SummaryCollector, SummaryObservationCollector
from ert_gui.plottery import PlotConfig, PlotContext, EnsemblePlot, ObservationPlot


class SummaryPlot(object):

    @staticmethod
    def summaryEnsemblePlot(plot_context):
        """
        @type plot_context: PlotContext
        """
        ert = plot_context.ert()
        key = plot_context.key()
        axes = plot_context.figure().add_subplot(111)
        """:type: matplotlib.axes.Axes """
        config = plot_context.plotConfig()

        if config.xLabel() is None:
            config.setXLabel("Date")

        if config.yLabel() is None:
            config.setYLabel("Value")

            if ert.eclConfig().hasRefcase():
                unit = ert.eclConfig().getRefcase().unit(key)
                if unit != "":
                    config.setYLabel(unit)

        axes.set_title(config.title())

        case_list = plot_context.cases()
        for case in case_list:
            data = SummaryCollector.loadAllSummaryData(ert, case, [key])
            if not data.empty:
                data = data.reset_index()
                data = data.pivot(index="Date", columns="Realization", values=key)

                EnsemblePlot.plot(axes, config, data, case)
                config.nextColor()

        if config.isObservationsEnabled() and ert.getKeyManager().isKeyWithObservations(key):
            if len(case_list) > 0:
                observation_data = SummaryObservationCollector.loadObservationData(ert, case_list[0], [key])

                if not observation_data.empty:
                    ObservationPlot.plot(axes, config, observation_data, value_column=key)


        if config.isRefcaseEnabled() and ert.eclConfig().hasRefcase():
            refcase_data = SummaryPlot.getRefcaseData(ert, key)

            if not refcase_data.empty:
                EnsemblePlot.plotRefcase(axes, config, refcase_data)


        axes.legend(config.legendItems(), config.legendLabels())
        axes.grid()

        plot_context.figure().autofmt_xdate()

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
