from ert._c_wrappers.enkf import EnkfConfigNode, EnkfNode, NodeId
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert._c_wrappers.test import ErtTestContext
from ert.simulator.simulation_context import _run_forward_model

from ...libres_utils import ResTest, tmpdir


class SimulationBatchTest(ResTest):
    def setUp(self):
        pass

    @tmpdir()
    def test_run(self):
        ens_size = 2
        config_file = self.createTestPath("local/config/simulation_batch/config.ert")
        with ErtTestContext(model_config=config_file) as ctx:
            ert = ctx.getErt()
            ens_config = ert.ensembleConfig()

            # Observe that a significant amount of hardcoding
            # regarding the GEN_DATA and EXT_PARAM nodes is assumed
            # between this test, the config file and the forward model.

            # Add control nodes
            order_control = EnkfConfigNode.create_ext_param(
                "WELL_ORDER", ["W1", "W2", "W3"]
            )
            injection_control = EnkfConfigNode.create_ext_param(
                "WELL_INJECTION", ["W1", "W4"]
            )
            ens_config.addNode(order_control)
            ens_config.addNode(injection_control)

            # Add result nodes
            order_result = EnkfConfigNode.create_gen_data("ORDER", "order_%d")
            injection_result = EnkfConfigNode.create_gen_data(
                "INJECTION", "injection_%d"
            )
            ens_config.addNode(order_result)
            ens_config.addNode(injection_result)

            order_node = EnkfNode(order_control)
            order_node_ext = order_node.as_ext_param()
            injection_node = EnkfNode(injection_control)
            injection_node_ext = injection_node.as_ext_param()

            fs_manager = ert.getEnkfFsManager()
            sim_fs = fs_manager.getFileSystem("sim_fs")
            state_map = sim_fs.getStateMap()
            batch_size = ens_size + 2
            for iens in range(batch_size):
                node_id = NodeId(0, iens)

                order_node_ext["W1"] = iens
                order_node_ext["W2"] = iens * 10
                order_node_ext["W3"] = iens * 100
                order_node.save(sim_fs, node_id)

                injection_node_ext["W1"] = iens + 1
                injection_node_ext["W4"] = 3 * (iens + 1)
                injection_node.save(sim_fs, node_id)
                state_map[iens] = RealizationStateEnum.STATE_INITIALIZED

            mask = [True] * batch_size
            run_context = ert.create_ensemble_experiment_run_context(
                source_filesystem=sim_fs, active_mask=mask, iteration=0
            )
            ert.createRunPath(run_context)
            job_queue = ert.get_queue_config().create_job_queue()

            ert.createRunPath(run_context)
            num = _run_forward_model(ert, job_queue, run_context)
            self.assertEqual(num, batch_size)

            order_result = EnkfNode(ens_config["ORDER"])
            injection_result = EnkfNode(ens_config["INJECTION"])

            for iens in range(batch_size):
                node_id = NodeId(0, iens)
                order_result.load(sim_fs, node_id)
                data = order_result.asGenData()

                order_node.load(sim_fs, node_id)
                self.assertEqual(order_node_ext["W1"], data[0])
                self.assertEqual(order_node_ext["W2"], data[1])
                self.assertEqual(order_node_ext["W3"], data[2])
