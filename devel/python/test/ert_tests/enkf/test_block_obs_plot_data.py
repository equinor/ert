#!/usr/bin/env python
#  Copyright (C) 2013  Statoil ASA, Norway.
#
#  The file 'test_analysis_config.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
from ert.ecl import EclGrid
from ert.enkf.data import EnkfConfigNode
from ert.enkf.ensemble_data import PlotBlockData
from ert.enkf.ensemble_data.plot_block_vector import PlotBlockVector
from ert.enkf.enums import EnkfTruncationType, EnkfObservationImplementationType
from ert.enkf.observations import ObsVector

from ert_tests import ExtendedTestCase


class BlockObsPlotDataTest(ExtendedTestCase):

    def test_plot_block_data(self):
        grid = EclGrid.create_rectangular((10, 10, 10), (1, 1, 1))
        config_node = EnkfConfigNode.createFieldConfigNode("FIELD", grid)
        config_node.updateStateField(EnkfTruncationType.TRUNCATE_NONE, 0, 0)

        obs_vector = ObsVector(EnkfObservationImplementationType.BLOCK_OBS, "OBS", config_node, 100)

        pbd = PlotBlockData(obs_vector)
        self.assertEqual(len(pbd), 0)


    def test_plot_block_data_vector(self):
        grid = EclGrid.create_rectangular((10, 10, 10), (1, 1, 1))
        config_node = EnkfConfigNode.createFieldConfigNode("FIELD", grid)
        config_node.updateStateField(EnkfTruncationType.TRUNCATE_NONE, 0, 0)

        obs_vector = ObsVector(EnkfObservationImplementationType.BLOCK_OBS, "OBS", config_node, 100)

        pbv = PlotBlockVector(obs_vector, 0)
        self.assertEqual(len(pbv), 0)

