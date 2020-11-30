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

import os.path

from tests import ResTest
from tests.utils import tmpdir

from res.test import ErtTestContext
from res.enkf import ESUpdate
from res.enkf import ErtRunContext
from res.enkf.local_ministep import LocalMinistep
from res.enkf.local_obsdata import LocalObsdata
from res.enkf.local_updatestep import LocalUpdateStep
from res.enkf.local_obsdata_node import LocalObsdataNode


class LocalConfigTest(ResTest):
    def setUp(self):
        self.config = self.createTestPath("local/mini_ert/mini_config")
        self.local_conf_path = "python/enkf/data/local_config"

    def test_write_summary(self):
        with ErtTestContext(self.local_conf_path, self.config) as test_context:
            main = test_context.getErt()

            local_config = main.getLocalConfig()

            fname = "local_config_summary.txt"
            local_config.writeSummaryFile(fname)
            self.assertTrue(os.path.isfile(fname))

    def test_get_grid(self):
        with ErtTestContext(self.local_conf_path, self.config) as test_context:
            main = test_context.getErt()
            local_config = main.getLocalConfig()
            grid = local_config.getGrid()
            dimens = grid.getNX(), grid.getNY(), grid.getNZ()
            self.assertEqual((10, 10, 5), dimens)

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

            # Try to add existing obsdata
            with self.assertRaises(ValueError):
                local_config.createObsdata("OBSSET_1")

            # Add node with range
            with self.assertRaises(KeyError):
                local_obs_data_1.addNodeAndRange("MISSING_KEY", 0, 1)

            local_obs_data_1.addNodeAndRange("GEN_PERLIN_1", 0, 1)
            local_obs_data_1.addNodeAndRange("GEN_PERLIN_2", 0, 1)
            self.assertEqual(len(local_obs_data_1), 2)

            # Delete node
            del local_obs_data_1["GEN_PERLIN_1"]
            self.assertEqual(len(local_obs_data_1), 1)

            # Get node
            node = local_obs_data_1["GEN_PERLIN_2"]
            self.assertTrue(isinstance(node, LocalObsdataNode))

            # Add node again with no range and check return type
            node_again = local_obs_data_1.addNode("GEN_PERLIN_1")
            self.assertTrue(isinstance(node_again, LocalObsdataNode))

            # Error when adding existing obs node
            with self.assertRaises(KeyError):
                local_obs_data_1.addNode("GEN_PERLIN_1")

    def test_attach_obs_data(self):
        with ErtTestContext(self.local_conf_path, self.config) as test_context:
            main = test_context.getErt()

            local_config = main.getLocalConfig()
            local_obs_data_2 = local_config.createObsdata("OBSSET_2")
            self.assertTrue(isinstance(local_obs_data_2, LocalObsdata))

            # Obsdata
            local_obs_data_2.addNodeAndRange("GEN_PERLIN_1", 0, 1)
            local_obs_data_2.addNodeAndRange("GEN_PERLIN_2", 0, 1)

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
            self.assertEqual(1, len(ministep))
            dataset = ministep["ALL_DATA"]
            self.assertTrue("PERLIN_PARAM" in dataset)

            obsdata = ministep.getLocalObsData()
            self.assertEqual(len(obsdata), 3)

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

            self.assertFalse("DATA" in ministep)
            with self.assertRaises(KeyError):
                _ = ministep["DATA"]

            self.assertIsNone(ministep.get_obs_data())

    def test_attach_ministep(self):
        with ErtTestContext(self.local_conf_path, self.config) as test_context:
            main = test_context.getErt()

            local_config = main.getLocalConfig()

            # Update step
            updatestep = local_config.getUpdatestep()
            self.assertTrue(isinstance(updatestep, LocalUpdateStep))
            upd_size = len(updatestep)

            # Ministep
            ministep = local_config.createMinistep("MINISTEP")
            self.assertTrue(isinstance(ministep, LocalMinistep))

            # Attach
            updatestep.attachMinistep(ministep)
            self.assertTrue(isinstance(updatestep[0], LocalMinistep))
            self.assertEqual(len(updatestep), upd_size + 1)

    @tmpdir()
    def test_attach_obs_data_to_ministep(self):
        config = self.createTestPath("local/snake_oil/snake_oil.ert")

        expected_keys = {
            "WPR_DIFF_1",
            "WOPR_OP1_108",
            "FOPR",
            "WOPR_OP1_144",
            "WOPR_OP1_190",
            "WOPR_OP1_9",
            "WOPR_OP1_36",
            "WOPR_OP1_72",
        }

        with ErtTestContext("obs_data_ministep_test", config) as context:
            ert = context.getErt()
            es_update = ESUpdate(ert)
            fsm = ert.getEnkfFsManager()

            sim_fs = fsm.getFileSystem("default_0")
            target_fs = fsm.getFileSystem("target")
            run_context = ErtRunContext.ensemble_smoother_update(sim_fs, target_fs)
            es_update.smootherUpdate(run_context)

            update_step = ert.getLocalConfig().getUpdatestep()
            ministep = update_step[len(update_step) - 1]
            obs_data = ministep.get_obs_data()
            self.assertEqual(len(expected_keys), obs_data.get_num_blocks())

            observed_obs_keys = set()
            for block_num in range(obs_data.get_num_blocks()):
                block = obs_data.get_block(block_num)

                obs_key = block.get_obs_key()
                observed_obs_keys.add(obs_key)
                for i in range(len(block)):
                    self.assertTrue(block.is_active(i))

            self.assertSetEqual(expected_keys, observed_obs_keys)
