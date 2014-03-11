# Copyright (C) 2013  Statoil ASA, Norway.
#
# The file 'ensemble_block_data_fetcher.py' is part of ERT - Ensemble based Reservoir Tool.
#
# ERT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ERT is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
# for more details.
from ert.enkf.ensemble_data import PlotBlockData
from ert.enkf.enums import ErtImplType
from ert.enkf.plot import DataFetcher


class EnsembleBlockDataFetcher(DataFetcher):
    def __init__(self, ert):
        super(EnsembleBlockDataFetcher, self).__init__(ert)

    def __fetchSimulationData(self, block_data):
        data = {
            "x": [],
            "y": [],
            "min_x_values": [],
            "max_x_values": [],
            "min_y": None,
            "max_y": None,
            "min_x": None,
            "max_x": None
        }

        depth_vector = block_data.getDepth()

        for depth in depth_vector:
            data["y"].append(depth)
            data["min_x_values"].append(None)
            data["max_x_values"].append(None)


        min_y = min(data["y"])
        max_y = max(data["y"])

        if data["min_y"] is None or data["min_y"] > min_y:
            data["min_y"] = min_y

        if data["max_y"] is None or data["max_y"] < max_y:
            data["max_y"] = max_y



        for block_vector in block_data:
            x = []
            data["x"].append(x)

            for index in range(len(block_vector)):
                value = block_vector[index]
                x.append(value)
                if data["min_x"] is None or data["min_x"] > value:
                    data["min_x"] = value

                if data["max_x"] is None or data["max_x"] < value:
                    data["max_x"] = value

                if data["min_x_values"][index] is None or data["min_x_values"][index] > value:
                    data["min_x_values"][index] = value

                if data["max_x_values"][index] is None or data["max_x_values"][index] < value:
                    data["max_x_values"][index] = value
        return data

    def fetchData(self, key, case=None):
        enkf_fs = self.ert().getEnkfFsManager().getFileSystem(case, read_only = True)
        observations = self.ert().getObservations()
        assert observations.hasKey(key)

        observation_vector = observations.getObservationsVector(key)

        report_step_data = []
        for report_step in observation_vector:
            block_data = PlotBlockData(observation_vector, enkf_fs, report_step)
            data = self.__fetchSimulationData(block_data)
            data["report_step"] = report_step

            report_step_data.append(data)


        return report_step_data

    def fetchSupportedKeys(self):
        string_list = self.ert().ensembleConfig().getKeylistFromImplType(ErtImplType.SUMMARY)
        return [key for key in string_list]

