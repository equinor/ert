#  Copyright (C) 2017  Equinor ASA, Norway.
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


from libres_utils import ResTest

from res.enkf import LocalObsdata, LocalConfig
from res.enkf.active_list import ActiveList
from res.enkf.enums import ActiveMode
from res.enkf.local_ministep import LocalMinistep
from res.enkf.local_obsdata import LocalObsdataNode
from res.test import ErtTestContext


class LocalConfigTest(ResTest):
    def setUp(self):
        self.config = self.createTestPath("local/snake_oil_field/snake_oil.ert")
        self.local_conf_path = "python/enkf/data/local_config"

    def test_local_obs_data(self):
        with ErtTestContext(self.local_conf_path, self.config) as test_context:
            main = test_context.getErt()
            self.assertTrue(main, msg="Load failed")

            local_config = main.getLocalConfig()

            local_config.clear()
            updatestep = local_config.getUpdatestep()
            self.assertEqual(0, len(updatestep))

            # Creating obsdata
            local_obs_data_1 = local_config.createObsdata("OBSSET_1")
            self.assertTrue(isinstance(local_obs_data_1, LocalObsdata))

            local_obs_data_1.addNode("GEN_PERLIN_1")
            local_obs_data_1.addNode("GEN_PERLIN_2")
            self.assertEqual(len(local_obs_data_1), 2)

            # Delete node
            del local_obs_data_1["GEN_PERLIN_1"]
            self.assertEqual(len(local_obs_data_1), 1)

            # Get node
            node = local_obs_data_1["GEN_PERLIN_2"]
            self.assertTrue(isinstance(node, LocalObsdataNode))

            # Add node again with no range and check return type
            node_added = local_obs_data_1.addNode("GEN_PERLIN_1")
            self.assertTrue(node_added)

            # Error when adding existing obs node
            with self.assertRaises(KeyError):
                local_obs_data_1.addNode("GEN_PERLIN_1")

            al = local_obs_data_1.getActiveList("GEN_PERLIN_1")
            al.addActiveIndex(10)
            self.assertEqual(al.getMode(), ActiveMode.PARTLY_ACTIVE)

    def test_get_active_list(self):
        with ErtTestContext(self.local_conf_path, self.config) as test_context:
            main = test_context.getErt()

            local_config = main.getLocalConfig()

            local_config.clear()
            local_obs_data_1 = local_config.createObsdata("OBSSET_1")
            local_obs_data_1.addNode("GEN_PERLIN_1")
            l2 = local_obs_data_1.getActiveList("GEN_PERLIN_1")
            assert isinstance(l2, ActiveList)

    def test_attach_obs_data(self):
        with ErtTestContext(self.local_conf_path, self.config) as test_context:
            main = test_context.getErt()

            local_config = main.getLocalConfig()
            local_obs_data_2 = local_config.createObsdata("OBSSET_2")
            self.assertTrue(isinstance(local_obs_data_2, LocalObsdata))

            local_obs_data_2.addNode("GEN_PERLIN_1")
            local_obs_data_2.addNode("GEN_PERLIN_2")
            # Ministep
            ministep = local_config.createMinistep("MINISTEP")
            self.assertTrue(isinstance(ministep, LocalMinistep))

            # Attach obsset
            ministep.attachObsset(local_obs_data_2)

            # Retrieve attached obsset
            local_obs_data_new = ministep.getLocalObsData()
            self.assertEqual(len(local_obs_data_new), 2)

    def test_all_active(self):
        with ErtTestContext(self.local_conf_path, self.config) as test_context:
            main = test_context.getErt()

            local_config = main.getLocalConfig()
            updatestep = local_config.getUpdatestep()
            ministep = updatestep[0]
            self.assertEqual(3, ministep.numActiveData())
            self.assertTrue(ministep.hasActiveData("SNAKE_OIL_PARAM"))

            obsdata = ministep.getLocalObsData()
            self.assertEqual(len(obsdata), 8)

    def test_ministep(self):
        with ErtTestContext(
            "python/enkf/data/local_config", self.config
        ) as test_context:
            main = test_context.getErt()

            local_config = main.getLocalConfig()
            analysis_module = main.analysisConfig().getModule("STD_ENKF")

            # Ministep
            ministep = local_config.createMinistep("MINISTEP", analysis_module)
            self.assertTrue(isinstance(ministep, LocalMinistep))

            with self.assertRaises(KeyError):
                _ = local_config.createMinistep("MINISTEP", None)

            self.assertFalse(ministep.hasActiveData("DATA"))
            with self.assertRaises(KeyError):
                _ = ministep.getActiveList("DATA")

    def test_attach_ministep(self):
        with ErtTestContext(self.local_conf_path, self.config) as test_context:
            main = test_context.getErt()

            local_config = main.getLocalConfig()

            # Update step
            updatestep = local_config.getUpdatestep()
            self.assertTrue(isinstance(updatestep, LocalConfig))
            upd_size = len(updatestep)

            # Ministep
            ministep = local_config.createMinistep("MINISTEP")
            self.assertTrue(isinstance(ministep, LocalMinistep))

            # Attach
            updatestep.attachMinistep(ministep)
            self.assertTrue(isinstance(updatestep[0], LocalMinistep))
            self.assertEqual(len(updatestep), upd_size + 1)

    def test_local_obsdata_node(self):
        node = LocalObsdataNode("OBS_NODE")
        self.assertEqual(node.key(), "OBS_NODE")
        self.assertEqual(node.getKey(), "OBS_NODE")

        al = node.getActiveList()
        self.assertTrue(isinstance(al, ActiveList))
