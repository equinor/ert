import datetime
import os
import time
import warnings
import sys
from threading import Thread

from res.enkf import EnKFMain
from res.enkf.enums import RealizationStateEnum
from res.server import ErtRPCServer, ErtRPCClient

from ecl.test import ExtendedTestCase
from res.test import ErtTestContext, ErtTest
from tests.res.server import RPCServiceContext


def realizationIsInitialized(ert, case_name, realization_number):
    assert isinstance(ert, EnKFMain)
    fs = ert.getEnkfFsManager().getFileSystem(case_name)
    state_map = fs.getStateMap()
    state = state_map[realization_number]
    return state == RealizationStateEnum.STATE_INITIALIZED or state == RealizationStateEnum.STATE_HAS_DATA


class RPCServiceTest(ExtendedTestCase):
    def setUp(self):
        self.config = self.createTestPath("local/snake_oil_no_data/snake_oil.ert")


    def test_deprecated_server_creation(self):
        with ErtTestContext("ert/server/rpc/server", self.config) as test_context:
            ert = test_context.getErt()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                server = ErtRPCServer(ert.getUserConfigFile())
                self.assertTrue(len(w) > 0)
                self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

            self.assertIsNotNone(server.port)

            expected_config_file = os.path.join(
                                            test_context.getCwd(),
                                            os.path.basename(self.config)
                                            )
            self.assertEqual(expected_config_file, server._config_file)

            thread = Thread(name="ErtRPCServerTest")
            thread.run = server.start
            thread.start()

            server.stop()

    def test_invalid_server_creation(self):
        with ErtTestContext("ert/server/rpc/server", self.config) as test_context:

            # No such file
            with self.assertRaises(IOError):
                server = ErtRPCServer("this/is/not/a/file")

            # No configuration
            with self.assertRaises(ValueError):
                server = ErtRPCServer(None)

            # Wrong enkf_main type
            with self.assertRaises(TypeError):
                enkf_main = test_context.getErt()
                server = ErtRPCServer(enkf_main=enkf_main.resConfig())


    def test_server_creation(self):
        with ErtTestContext("ert/server/rpc/server", self.config) as test_context:
            ert = test_context.getErt()
            server = ErtRPCServer(ert)

            self.assertIsNotNone(server.port)
            self.assertEqual(ert, server._enkf_main)

            expected_config_file = os.path.join(
                                            test_context.getCwd(),
                                            os.path.basename(self.config)
                                            )
            self.assertEqual(expected_config_file, server._config_file)

            thread = Thread(name="ErtRPCServerTest")
            thread.run = server.start
            thread.start()

            server.stop()

    
    def test_multiple_threads(self):
        expected_ckw = {
            0:{'RATING': 'EXCELLENT', 'NPV': 125692.534209},
            1:{'RATING': 'AVERAGE', 'NPV': 87384.4316741},
            2:{'RATING': 'GOOD', 'NPV': 113181.024141},
            3:{'RATING': 'AVERAGE', 'NPV': 91659.8650599},
            4:{'RATING': 'EXCELLENT', 'NPV': 134891.570703},
            5:{'RATING': 'GOOD', 'NPV': 117270.977546},
            6:{'RATING': 'GOOD', 'NPV': 106838.28455},
            7:{'RATING': 'EXCELLENT', 'NPV': 144001.339},
            8:{'RATING': 'AVERAGE', 'NPV': 95423.9155004},
            9:{'RATING': 'AVERAGE', 'NPV': 96123.0227439}
        }

        with RPCServiceContext("ert/server/rpc/multi_client", self.config, store_area=True) as server:
            client_count = len(expected_ckw)

            # initializeCase(server.ert, "default", 1)
            thread_success_state = {}

            # This is one-realisation at-a-time construction, which
            # does not really blend in with the run_context concept.
            def createClientInteractor(target_case_name, iens):
                def clientInteraction():
                    thread_success_state[iens] = False
                    keywords = {"SNAKE_OIL_PARAM": [0.50, iens + 2, 1.750, 0.250, 0.990, 2 + client_count - iens, 1.770, 0.330, 0.550, 0.770]}

                    client = ErtRPCClient("localhost", server.port)
                    self.assertTrue(client.isRunning())
                    self.assertTrue(client.isInitializationCaseAvailable())

                    client.addSimulation(0, 0, iens, keywords)
                    self.assertTrue(realizationIsInitialized(server.ert, target_case_name, iens))

                    while not client.isRealizationFinished(iens):
                        time.sleep(0.5)

                    self.assertTrue(client.didRealizationSucceed(iens))

                    result = client.getCustomKWResult(target_case_name, iens, "SNAKE_OIL_NPV")
                    self.assertTrue("NPV" in result)
                    self.assertTrue("RATING" in result)

                    self.assertEqual(expected_ckw[iens]["RATING"], result["RATING"])
                    self.assertAlmostEqual(expected_ckw[iens]["NPV"], result["NPV"])

                    thread_success_state[iens] = True

                return clientInteraction

            threads = []

            target_case_name = "default_1"
            client = ErtRPCClient("localhost", server.port)
            client.startSimulationBatch(target_case_name, target_case_name , client_count)
            
            for iens in range(client_count):
                thread = Thread(name="client_%d" % iens)
                thread.run = createClientInteractor(target_case_name, iens)
                threads.append(thread)

            for thread in threads:
                thread.start()

            while server.isRunning():
                time.sleep(0.1)

            for thread in threads:
                thread.join()

            self.assertTrue(all(success for success in thread_success_state.values()))


    def test_runtime_geoid(self):
        config_rel_path = "local/snake_oil_no_data/snake_oil_GEO_ID.ert"
        geoid_config_path = self.createTestPath(config_rel_path)

        runpath_root_fmt = "simulations/%s"
        for geo_id in range(3):
            with RPCServiceContext("ert/server/rpc/geoid", geoid_config_path, store_area=True) as server:
                server.startSimulationBatch("testing_geoid_%d" % geo_id,
                                            "results_geoid_%d" % geo_id,
                                            1)

                keywords = {"SNAKE_OIL_PARAM" : 10*[0.5]}
                server.addSimulation(geo_id, 0, 0, keywords)

                resolved_path = runpath_root_fmt % geo_id
                self.assertTrue(os.path.isdir(resolved_path))

                # TODO: This fails currently due to runpath creation both by
                # init of simulation_context and when adding simulation to the
                # ertrpcserver. Its the first one that is problematic
                #unresolved_path = runpath_root_fmt % '<GEO_ID>'
                #self.assertFalse(os.path.isdir(unresolved_path))
