from ert.enkf import EnsConfig
from ert.enkf.ensemble_data import EnsemblePlotData
from ert.enkf.enums import ErtImplType
from ert.enkf.plot.data_fetcher import DataFetcher
from ert_gui.time_it import timeit


class EnsembleDataFetcher(DataFetcher):
    def __init__(self, ert):
        super(EnsembleDataFetcher, self).__init__(ert)


    def getSummaryKeys(self):
        """ @rtype: StringList """
        return self.ert().ensembleConfig().getKeylistFromImplType(ErtImplType.SUMMARY)


    def getEnsembleConfigNode(self, key):
        """ @rtype: EnsConfig """
        ensemble_config = self.ert().ensembleConfig()
        assert ensemble_config.hasKey(key)
        return ensemble_config.getNode(key)


    # @timeit
    def fetchData(self, key, case=None):
        ensemble_config_node = self.getEnsembleConfigNode(key)
        enkf_fs = self.ert().getEnkfFsManager().mountAlternativeFileSystem(case, True, False)
        ensemble_plot_data = EnsemblePlotData(ensemble_config_node, enkf_fs)

        data = {
            "x": [],
            "y": [],
            "min_y_values": None,
            "max_y_values": None,
            "min_y": None,
            "max_y": None,
            "min_x": None,
            "max_x": None
        }

        for vector in ensemble_plot_data:
            if data["x"] is None or data["y"] is None:
                data["x"] = []
                data["y"] = []

            if data["min_y_values"] is None or data["max_y_values"] is None:
                data["min_y_values"] = []
                data["max_y_values"] = []

                for index in range(len(vector)):
                    if vector.isActive(index):
                        data["min_y_values"].append(vector.getValue(index))
                        data["max_y_values"].append(vector.getValue(index))


            x = []
            y = []
            data["x"].append(x)
            data["y"].append(y)

            active_index = 0
            for index in range(len(vector)):
                if vector.isActive(index):
                    y_value = vector.getValue(index)
                    y.append(y_value)

                    x_value = vector.getTime(index).ctime()
                    x.append(x_value)

                    if data["min_x"] is None or data["min_x"] > x_value:
                        data["min_x"] = x_value

                    if data["max_x"] is None or data["max_x"] < x_value:
                        data["max_x"] = x_value

                    if data["min_y"] is None or data["min_y"] > y_value:
                        data["min_y"] = y_value

                    if data["max_y"] is None or data["max_y"] < y_value:
                        data["max_y"] = y_value


                    if data["min_y_values"][active_index] is None or data["min_y_values"][active_index] > y_value:
                        data["min_y_values"][active_index] = y_value

                    if data["max_y_values"][active_index] is None or data["max_y_values"][active_index] < y_value:
                        data["max_y_values"][active_index] = y_value


                    active_index += 1

        return data