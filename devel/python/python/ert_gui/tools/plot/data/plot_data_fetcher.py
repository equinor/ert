from ert.enkf.plot import EnsembleDataFetcher, ObservationDataFetcher, RefcaseDataFetcher
from ert_gui.tools.plot.data import PlotData, ObservationPlotData, EnsemblePlotData, RefcasePlotData
from ert_gui.models import ErtConnector
from ert_gui.models.mixins import ModelMixin


class PlotDataFetcher(ErtConnector, ModelMixin):

    def getPlotDataForKeyAndCases(self, key, cases):
        plot_data = PlotData(key)

        observation_data = ObservationDataFetcher(self.ert()).fetchData(key)
        observation_plot_data = ObservationPlotData(key)
        observation_plot_data.setObservationData(observation_data["x"], observation_data["y"], observation_data["std"], observation_data["continuous"])
        observation_plot_data.updateBoundaries(observation_data["min_x"], observation_data["max_x"], observation_data["min_y"], observation_data["max_y"])

        plot_data.setObservationData(observation_plot_data)


        refcase_data = RefcaseDataFetcher(self.ert()).fetchData(key)
        refcase_plot_data = RefcasePlotData(key)
        refcase_plot_data.setRefcaseData(refcase_data["x"], refcase_data["y"])
        refcase_plot_data.updateBoundaries(refcase_data["min_x"], refcase_data["max_x"], refcase_data["min_y"], refcase_data["max_y"])

        plot_data.setRefcaseData(refcase_plot_data)


        for case in cases:
            ensemble_data = EnsembleDataFetcher(self.ert()).fetchData(key, case)

            ensemble_plot_data = EnsemblePlotData(key, case)
            ensemble_plot_data.setEnsembleData(ensemble_data["x"], ensemble_data["y"], ensemble_data["min_y_values"], ensemble_data["max_y_values"])
            ensemble_plot_data.updateBoundaries(ensemble_data["min_x"], ensemble_data["max_x"], ensemble_data["min_y"], ensemble_data["max_y"])
            plot_data.addEnsembleData(ensemble_plot_data)


        return plot_data
