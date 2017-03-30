#  Copyright (C) 2017  Statoil ASA, Norway.
#
#  This file is part of ERT - Ensemble based Reservoir Tool.
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

import os.path

from ert.test import ExtendedTestCase
from ert.test import ErtTestContext
from ert.enkf.active_list import ActiveList
from ert.enkf.local_dataset import LocalDataset

from ert.ecl import EclRegion
from ert.geo import Surface, GeoRegion

class LocalDatasetTest(ExtendedTestCase):

    def setUp(self):
        self.config = self.createTestPath('local/custom_kw/mini_config')
        self.field_config = self.createTestPath('local/snake_oil_field/snake_oil.ert')
        self.surf_config = self.createTestPath('local/snake_oil_field/snake_oil_surface.ert')

    def _small_surf(self):
        # values copied from irap surface_small
        ny,nx = 20,30
        xinc,yinc = 50.0, 50.0
        xstart,ystart = 463325.5625, 7336963.5
        angle = -65.0
        s_args = (None, nx, ny, xinc, yinc, xstart, ystart, angle)
        return Surface(*s_args)

    def test_local_field(self):
        with ErtTestContext('python/enkf/data/local_config', self.field_config) as test_context:
            main = test_context.getErt()
            local_config = main.getLocalConfig()

            # Creating dataset
            data_scale = local_config.createDataset('DATA_SCALE')
            grid = local_config.getGrid()
            ecl_reg = EclRegion(grid, False)
            ecl_reg.select_islice(10, 20)
            data_scale.addField('PERMX', ecl_reg)
            self.assertEqual(1, len(data_scale))


    def test_local_surface(self):
        with ErtTestContext('python/enkf/data/local_config', self.surf_config) as test_context:
            main = test_context.getErt()
            local_config = main.getLocalConfig()

            # Creating dataset
            data_scale = local_config.createDataset('DATA_SCALE')
            surf = self._small_surf()
            ps = surf.getPointset()
            geo_reg = GeoRegion(ps)
            data_scale.addSurface('TOP', geo_reg)
            self.assertEqual(1, len(data_scale))

    def test_local_dataset(self):
        with ErtTestContext('python/enkf/data/local_config', self.config) as test_context:
            main = test_context.getErt()

            local_config = main.getLocalConfig()

            # Creating dataset
            data_scale = local_config.createDataset('DATA_SCALE')
            self.assertTrue(isinstance(data_scale, LocalDataset))

            # Try to add existing dataset
            with self.assertRaises(ValueError):
                local_config.createDataset('DATA_SCALE')

            with self.assertRaises(KeyError):
                data_scale.addNode('KEY_NOTIN_ENSEMBLE')

            self.assertEqual(0, len(data_scale))
            data_scale.addNode('PERLIN_PARAM')
            self.assertEqual(1, len(data_scale))
            active_list = data_scale.getActiveList('PERLIN_PARAM')
            self.assertTrue(isinstance(active_list, ActiveList))
            active_list.addActiveIndex(0)
            self.assertTrue('PERLIN_PARAM' in data_scale)
            self.assertFalse('KEY_NOTIN_ENSEMBLE' in data_scale)

            ministep = local_config.createMinistep('MINISTEP')
            ministep.attachDataset(data_scale)
            self.assertTrue('DATA_SCALE' in ministep)
            data_scale_get = ministep['DATA_SCALE']
            self.assertTrue('PERLIN_PARAM' in data_scale_get)

            # Error when adding existing data node
            with self.assertRaises(KeyError):
                data_scale.addNode('PERLIN_PARAM')
