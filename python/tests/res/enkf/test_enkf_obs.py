import pytest

from tests import ResTest
from tests.utils import tmpdir
from res.test import ErtTestContext

from ecl.grid import EclGrid
from ecl.summary import EclSum
from res.sched import History

from ecl.util.util import BoolVector,IntVector
from res.enkf import ActiveMode, EnsembleConfig
from res.enkf import (ObsVector, LocalObsdata, EnkfObs, TimeMap,
                      LocalObsdataNode, ObsData, MeasData, ActiveList)

@pytest.mark.equinor_test
class EnKFObsTest(ResTest):
    def setUp(self):
        self.config_file = self.createTestPath("Equinor/config/obs_testing/config")
        self.obs_config = self.createTestPath("Equinor/config/obs_testing/observations")
        self.obs_config2 = self.createTestPath("Equinor/config/obs_testing/observations2")
        self.refcase = self.createTestPath("Equinor/config/obs_testing/EXAMPLE_01_BASE")
        self.grid = self.createTestPath("Equinor/config/obs_testing/EXAMPLE_01_BASE.EGRID")

    @tmpdir()
    def test_scale_obs(self):
        with ErtTestContext("obs_test", self.config_file) as test_context:
            ert = test_context.getErt()
            obs = ert.getObservations()

            obs1 = obs["WWCT:OP_1"].getNode(50)
            obs2 = obs["WWCT:OP_1_50"].getNode(50)

            self.assertEqual(obs1.getStandardDeviation(), obs2.getStandardDeviation())
            std0 = obs1.getStandardDeviation()

            local_obsdata = LocalObsdata("obs", obs)
            node1 = local_obsdata.addNode("WWCT:OP_1")
            node2 = local_obsdata.addNode("WWCT:OP_1_50")
            node1.addTimeStep(50)
            node2.addTimeStep(50)

            mask = BoolVector(default_value=True)
            mask[2] = True
            meas_data = MeasData(mask)
            obs_data = ObsData()
            fs = ert.getEnkfFsManager().getCurrentFileSystem()
            active_list = IntVector()
            active_list.initRange(0,2,1)
            obs.getObservationAndMeasureData(fs, local_obsdata, active_list, meas_data, obs_data)
            self.assertEqual(2, len(obs_data))

            v1 = obs_data[0]
            v2 = obs_data[1]

            self.assertEqual(v1[1], std0)
            self.assertEqual(v2[1], std0)

            meas_data = MeasData(mask)
            obs_data = ObsData(10)
            obs.getObservationAndMeasureData(fs, local_obsdata, active_list, meas_data, obs_data)
            self.assertEqual(2, len(obs_data))

            v1 = obs_data[0]
            v2 = obs_data[1]

            self.assertEqual(v1[1], std0*10)
            self.assertEqual(v2[1], std0*10)

            actl = ActiveList()
            obs1.updateStdScaling(10, actl)
            obs2.updateStdScaling(20, actl)
            meas_data = MeasData(mask)
            obs_data = ObsData()
            obs.getObservationAndMeasureData(fs, local_obsdata, active_list, meas_data, obs_data)
            self.assertEqual(2, len(obs_data))

            v1 = obs_data[0]
            v2 = obs_data[1]

            self.assertEqual(v1[1], std0*10)
            self.assertEqual(v2[1], std0*20)

    @tmpdir()
    def testObs(self):
        with ErtTestContext("obs_test", self.config_file) as test_context:
            ert = test_context.getErt()
            obs = ert.getObservations()

            self.assertEqual(32, len(obs))
            for v in obs:
                self.assertTrue(isinstance(v, ObsVector))

            self.assertEqual(obs[-1].getKey(),    'RFT_TEST')
            self.assertEqual(obs[-1].getDataKey(),'4289383')
            self.assertEqual(obs[-1].getObsKey(), 'RFT_TEST')

            with self.assertRaises(IndexError):
                v = obs[-40]
            with self.assertRaises(IndexError):
                v = obs[40]

            with self.assertRaises(KeyError):
                v = obs["No-this-does-not-exist"]

            v1 = obs["WWCT:OP_3"]
            v2 = obs["GOPT:OP"]
            mask = BoolVector(True, ert.getEnsembleSize())
            current_fs = ert.getEnkfFsManager().getCurrentFileSystem()

            self.assertTrue(v1.hasData(mask, current_fs))
            self.assertFalse(v2.hasData(mask, current_fs))

            local_node = v1.createLocalObs()
            for t in v1.getStepList():
                self.assertTrue(local_node.tstepActive(t))

    @tmpdir()
    def test_obs_block_scale_std(self):
        with ErtTestContext("obs_test_scale", self.config_file) as test_context:
            ert = test_context.getErt()
            fs = ert.getEnkfFsManager().getCurrentFileSystem()
            active_list = IntVector()
            active_list.initRange(0, ert.getEnsembleSize(), 1)

            obs = ert.getObservations()
            obs_data = LocalObsdata("OBSxx", obs)
            obs_vector = obs["WWCT:OP_1"]
            obs_data.addObsVector(obs_vector)
            scale_factor = obs.scaleCorrelatedStd(fs, obs_data, active_list)

            for obs_node in obs_vector:
                for index in range(len(obs_node)):
                    self.assertEqual(scale_factor, obs_node.getStdScaling(index))




    def test_obs_block_all_active_local(self):
        with ErtTestContext("obs_test_all_active", self.config_file) as test_context:
            ert = test_context.getErt()
            obs = ert.getObservations()
            obs_data = obs.getAllActiveLocalObsdata()

            self.assertEqual(len(obs_data), len(obs))
            for obs_vector in obs:
                self.assertIn(obs_vector.getObservationKey(), obs_data)

                tstep_list1 = obs_vector.getStepList()
                local_node = obs_data[ obs_vector.getObservationKey() ]
                for t in tstep_list1:
                    self.assertTrue(local_node.tstepActive(t))

                active_list = local_node.getActiveList()
                self.assertEqual(active_list.getMode(), ActiveMode.ALL_ACTIVE)



    def test_create(self):
        ensemble_config = EnsembleConfig()
        obs = EnkfObs(ensemble_config)
        self.assertEqual(len(obs), 0)

        self.assertFalse(obs.valid)
        with self.assertRaises(ValueError):
            obs.load(self.obs_config)
        self.assertEqual(len(obs), 0)


        time_map = TimeMap()
        obs = EnkfObs(ensemble_config, external_time_map=time_map)
        self.assertEqual(len(obs), 0)

        grid = EclGrid(self.grid)
        refcase = EclSum(self.refcase)

        history = History(refcase, False)
        obs = EnkfObs(ensemble_config, grid=grid, history=history)
        self.assertTrue(obs.valid)
        with self.assertRaises(IOError):
            obs.load("/does/not/exist")

        obs.load(self.obs_config)
        self.assertTrue(obs.valid)
        self.assertEqual(len(obs), 33)
        obs.clear()
        self.assertEqual(len(obs), 0)

        obs.load(self.obs_config)
        self.assertEqual(len(obs), 33)
        self.assertNotIn("RFT2", obs)
        obs.load(self.obs_config2)
        self.assertEqual(len(obs), 35)
        self.assertIn("RFT2", obs)

    def test_hookmanager_runpathlist(self):
        with ErtTestContext("obs_test", self.config_file) as test_context:
            ert = test_context.getErt()
            hm = ert.getHookManager()
            self.assertGreater(len(repr(hm)), 0)

            rpl = hm.getRunpathList()
            self.assertGreater(len(repr(rpl)), 0)

            ef = rpl.getExportFile()
            self.assertTrue('.ert_runpath_list' in ef)
            nf = 'myExportCamel'
            rpl.setExportFile('myExportCamel')
            ef = rpl.getExportFile()
            self.assertTrue(nf in ef)



    def test_ert_obs_reload(self):
        with ErtTestContext("obs_test_reload", self.config_file) as test_context:
            ert = test_context.getErt()
            local_config = ert.getLocalConfig()
            update_step = local_config.getUpdatestep()
            mini_step = update_step[0]
            local_obs = mini_step.getLocalObsData()
            self.assertTrue("WGOR:OP_5" in local_obs)
            self.assertTrue("RPR2_1" in local_obs)


            ens_config = ert.ensembleConfig()
            wwct_op1 = ens_config["WWCT:OP_1"]
            wopr_op5 = ens_config["WOPR:OP_5"]

            obs = ert.getObservations()
            self.assertEqual(len(obs), 32)

            keys = wwct_op1.getObservationKeys()
            self.assertEqual(len(keys), 2)
            self.assertTrue("WWCT:OP_1" in keys)
            self.assertTrue("WWCT:OP_1_50" in keys)

            self.assertEqual(wopr_op5.getObservationKeys(), [])

            ert.loadObservations("observations2")
            self.assertEqual(len(obs), 2)
            self.assertEqual(wwct_op1.getObservationKeys(), [])
            self.assertEqual(wopr_op5.getObservationKeys(), ["WOPR:OP_5"])

            local_config = ert.getLocalConfig()
            update_step = local_config.getUpdatestep()
            mini_step = update_step[0]
            local_obs = mini_step.getLocalObsData()
            self.assertTrue("WOPR:OP_5" in local_obs)
            self.assertTrue("RFT2" in local_obs)
            self.assertFalse("WGOR:OP_5" in local_obs)
            self.assertFalse("RPR2_1" in local_obs)


            ert.loadObservations("observations", clear=False)
            self.assertEqual(len(obs), 34)
            keys = wwct_op1.getObservationKeys()
            self.assertEqual(len(keys), 2)
            self.assertTrue("WWCT:OP_1" in keys)
            self.assertTrue("WWCT:OP_1_50" in keys)

            self.assertEqual(wopr_op5.getObservationKeys(), ["WOPR:OP_5"])
