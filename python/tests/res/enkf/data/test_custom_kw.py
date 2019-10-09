import os

import pytest

from res.enkf.enums import EnkfRunType
from res.enkf import ErtRunContext
from res.enkf.config import CustomKWConfig
from res.enkf.data import CustomKW
from res.enkf.enkf_simulation_runner import EnkfSimulationRunner
from res.enkf.export import custom_kw_collector
from res.enkf.export.custom_kw_collector import CustomKWCollector
from res.test.ert_test_context import ErtTestContext
from tests import ResTest
from ecl.util.test.test_area import TestAreaContext
from ecl.util.util import StringList
from ecl.util.util import BoolVector

from tests.utils import tmpdir

class CustomKWTest(ResTest):

    def createResultFile(self, filename, data):
        with open(filename, "w") as output_file:
            for key in data:
                output_file.write("%s %s\n" % (key, data[key]))

    def test_custom_kw_creation(self):
        data = {"VALUE_1": 2345.234,
                "VALUE_2": 0.001234,
                "VALUE_3": "string_1",
                "VALUE_4": "string_2"}

        with TestAreaContext("python/enkf/data/custom_kw_creation") as test_area:

            self.createResultFile("result_file", data)

            custom_kw_config = CustomKWConfig("CUSTOM_KW", "result_file")

            self.assertEqual(len(custom_kw_config), 0)

            custom_kw = CustomKW(custom_kw_config)

            custom_kw.fload("result_file")

            self.assertEqual(len(custom_kw_config), 4)

            for key in data:
                index = custom_kw_config.indexOfKey(key)
                self.assertEqual(data[key], custom_kw[key])

            with self.assertRaises(KeyError):
                value = custom_kw["VALUE_5"]



    def test_custom_kw_config_data_is_null(self):
            data_1 = {"VALUE_1": 123453.3,
                      "VALUE_2": 0.234234}

            data_2 = {"VALUE_1": 965689,
                      "VALUE_3": 1.1222}

            with TestAreaContext("python/enkf/data/custom_kw_null_element") as test_area:

                self.createResultFile("result_file_1", data_1)
                self.createResultFile("result_file_2", data_2)

                custom_kw_config = CustomKWConfig("CUSTOM_KW", "result_file")

                custom_kw_1 = CustomKW(custom_kw_config)
                custom_kw_1.fload("result_file_1")

                custom_kw_2 = CustomKW(custom_kw_config)
                custom_kw_2.fload("result_file_2")

                index_1 = custom_kw_config.indexOfKey("VALUE_1")
                index_2 = custom_kw_config.indexOfKey("VALUE_2")

                self.assertEqual(custom_kw_1["VALUE_1"], data_1["VALUE_1"])
                self.assertEqual(custom_kw_2["VALUE_1"], data_2["VALUE_1"])

                self.assertIsNone(custom_kw_2["VALUE_2"])
                self.assertFalse( "VALUE_3" in custom_kw_config )


    @tmpdir()
    def test_simulated_custom_kw(self):
        config = self.createTestPath("local/custom_kw/mini_config")
        with ErtTestContext("python/enkf/data/custom_kw_simulated", config) as context:
            ert = context.getErt()

            ensemble_config = ert.ensembleConfig()
            self.assertTrue("AGGREGATED" in ensemble_config)

            config = ensemble_config.getNode("AGGREGATED").getCustomKeywordModelConfig()

            self.assertEqual(len(config.getKeys()), 0)

            simulation_runner = EnkfSimulationRunner(ert)
            job_queue = ert.get_queue_config().create_job_queue()

            iteration_count = 0
            active = BoolVector(default_value = True, initial_size = 4)
            subst_list = ert.getDataKW( )
            runpath_fmt = ert.getModelConfig( ).getRunpathFormat( )
            fs_manager = ert.getEnkfFsManager( )
            fs = fs_manager.getFileSystem("fs")
            jobname_fmt = ert.getModelConfig( ).getJobnameFormat( )

            run_context = ErtRunContext( EnkfRunType.ENSEMBLE_EXPERIMENT , fs, None , active , runpath_fmt, jobname_fmt, subst_list , iteration_count)

            simulation_runner.createRunPath( run_context )
            simulation_runner.runEnsembleExperiment(job_queue, run_context)

            config = ensemble_config.getNode("AGGREGATED").getCustomKeywordModelConfig()

            self.assertEqual(len(config.getKeys()), 4)
            self.assertItemsEqual(config.getKeys(), ["PERLIN_1", "PERLIN_2", "PERLIN_3", "STATE"])


    def test_custom_kw_set_values(self):
        definition = {
            "STRING": str,
            "FLOAT": float,
            "INT": float
        }

        ckwc = CustomKWConfig("Test", None, definition=definition)

        ckw = CustomKW(ckwc)
        with self.assertRaises(KeyError):
            ckw["ANOTHER_STRING"] = "another string"

        ckw["STRING"] = "string"
        ckw["FLOAT"] = 3.1415
        ckw["INT"] = 1

        self.assertEqual(ckw["STRING"], "string")
        self.assertEqual(ckw["FLOAT"], 3.1415)
        self.assertEqual(ckw["INT"], 1)



