# Copyright (C) 2013  Statoil ASA, Norway.
#
# The file 'block_data_fetcher.py' is part of ERT - Ensemble based Reservoir Tool.
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
from ert.enkf import EnkfObservationImplementationType
from ert.enkf.ensemble_data import PlotBlockData
from ert.enkf.plot import DataFetcher


class BlockDataFetcher(DataFetcher):
    def __init__(self, ert):
        super(BlockDataFetcher, self).__init__(ert)

    def getBlockObsKeys(self, sort=True):
        observations = self.ert().getObservations()
        keys = observations.getTypedKeylist(EnkfObservationImplementationType.BLOCK_OBS)
        return sorted(keys) if sort else keys

    def isBlockObsKey(self, key):
        return key in self.getBlockObsKeys(sort=False)

    def __fetchData(self, block_data, report_step):
        data = {
            "report_step": report_step,
            "x": [],
            "y": [],
            "min_y": None,
            "max_y": None,
            "min_x": None,
            "max_x": None
        }

        depth_vector = block_data.getDepth()
        data["y"] = [value for value in depth_vector]
        data["min_y"] = min(data["y"])
        data["max_y"] = max(data["y"])

        for block_vector in block_data:
            x = []
            data["x"].append(x)
            for value in block_vector:
                x.append(value)

                if data["min_x"] is None or data["min_x"] > value:
                    data["min_x"] = value

                if data["max_x"] is None or data["max_x"] < value:
                    data["max_x"] = value

        return data

    def fetchData(self, key, case=None):
        enkf_fs = self.ert().getEnkfFsManager().mountAlternativeFileSystem(case, True, False)
        observations = self.ert().getObservations()
        assert observations.hasKey(key)

        observation_data = observations.getObservationsVector(key)

        data = []
        for report_step in observation_data:
            block_data = PlotBlockData(observation_data, enkf_fs, report_step)
            data.append(self.__fetchData(block_data, report_step))


        return data

