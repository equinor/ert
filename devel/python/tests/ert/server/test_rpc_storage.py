from random import random

from ert.enkf.enums import ErtImplType
from ert.enkf.export.custom_kw_collector import CustomKWCollector
from ert.server import ErtRPCClient
from ert.test import ExtendedTestCase
from tests.ert.server import RPCServiceContext, initializeCase


class RPCStorageTest(ExtendedTestCase):
    def test_rpc_storage(self):
        config = self.createTestPath("local/snake_oil_no_data/snake_oil.ert")

        with RPCServiceContext("ert/server/rpc/client_storage", config) as server:
            client = ErtRPCClient("localhost", server.port)

            group_name = "Test"
            storage_definition = {
                "PI": float,
                "DakotaVersion": str,
                "Gradient": float,
                "GradientDirection": str
            }
            client.prototypeStorage(group_name, storage_definition)

            with self.assertRaises(UserWarning):
                client.prototypeStorage(group_name, {"value": float})

            with self.assertRaises(TypeError):
                client.prototypeStorage(group_name, {"value": bool})


            ensemble_config = server.ert.ensembleConfig()

            self.assertIn(group_name, ensemble_config.getKeylistFromImplType(ErtImplType.CUSTOM_KW))

            custom_kw_config = ensemble_config.getNode(group_name).getCustomKeywordModelConfig()

            self.assertIn("PI", custom_kw_config)
            self.assertTrue(custom_kw_config.keyIsDouble("PI"))

            self.assertIn("DakotaVersion", custom_kw_config)
            self.assertFalse(custom_kw_config.keyIsDouble("DakotaVersion"))

            self.assertIn("Gradient", custom_kw_config)
            self.assertTrue(custom_kw_config.keyIsDouble("PI"))

            self.assertIn("GradientDirection", custom_kw_config)
            self.assertFalse(custom_kw_config.keyIsDouble("GradientDirection"))


            simulation_count = 10
            initializeCase(server.ert, "default", simulation_count)

            client.storeGlobalData("default", group_name, "PI", 3.1415)
            client.storeGlobalData("default", group_name, "DakotaVersion", "DAKOTA 6.2.0")

            gradients = [random() * 20.0 - 10.0 for _ in range(simulation_count)]
            gradient_directions = [("POSITIVE" if gradient >= 0.0 else "NEGATIVE") for gradient in gradients]
            for sim_id in range(simulation_count):
                gradient = gradients[sim_id]
                gradient_direction = gradient_directions[sim_id]
                client.storeSimulationData("default", group_name, "Gradient", gradient, sim_id)
                client.storeSimulationData("default", group_name, "GradientDirection", gradient_direction, sim_id)


            data = CustomKWCollector.loadAllCustomKWData(server.ert, "default")
            for sim_id in range(simulation_count):
                self.assertEqual(data["Test:PI"][sim_id], 3.1415)
                self.assertEqual(data["Test:DakotaVersion"][sim_id], "DAKOTA 6.2.0")
                self.assertEqual(data["Test:Gradient"][sim_id], gradients[sim_id])
                self.assertEqual(data["Test:GradientDirection"][sim_id], gradient_directions[sim_id])





