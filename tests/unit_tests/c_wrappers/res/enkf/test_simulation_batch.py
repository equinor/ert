import pytest

from ert._c_wrappers.enkf import EnkfConfigNode, EnKFMain, EnkfNode, NodeId
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert.simulator.simulation_context import _run_forward_model


@pytest.mark.integration_test
def test_run_simulation_batch(setup_case):
    ert = EnKFMain(setup_case("config/simulation_batch", "config.ert"))
    ens_size = 4
    ens_config = ert.ensembleConfig()

    # Observe that a significant amount of hardcoding
    # regarding the GEN_DATA and EXT_PARAM nodes is assumed
    # between this test, the config file and the forward model.

    # Add control nodes
    order_control = EnkfConfigNode.create_ext_param("WELL_ORDER", ["W1", "W2", "W3"])
    injection_control = EnkfConfigNode.create_ext_param("WELL_INJECTION", ["W1", "W4"])
    ens_config.addNode(order_control)
    ens_config.addNode(injection_control)

    # Add result nodes
    order_result = EnkfConfigNode.create_gen_data("ORDER", "order_%d")
    injection_result = EnkfConfigNode.create_gen_data("INJECTION", "injection_%d")
    ens_config.addNode(order_result)
    ens_config.addNode(injection_result)

    order_node = EnkfNode(order_control)
    order_node_ext = order_node.as_ext_param()
    injection_node = EnkfNode(injection_control)
    injection_node_ext = injection_node.as_ext_param()

    fs_manager = ert.storage_manager
    sim_fs = fs_manager.add_case("sim_fs")
    state_map = sim_fs.getStateMap()
    batch_size = ens_size
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
    run_context = ert.load_ensemble_context(sim_fs.case_name, mask, iteration=0)
    ert.createRunPath(run_context)
    job_queue = ert.get_queue_config().create_job_queue()

    ert.createRunPath(run_context)
    num = _run_forward_model(ert, job_queue, run_context)
    assert num == batch_size

    for iens in range(batch_size):
        node_id = NodeId(0, iens)
        data, _ = sim_fs.load_gen_data("ORDER-0", [iens])
        data = data.flatten()

        order_node.load(sim_fs, node_id)
        assert order_node_ext["W1"] == data[0]
        assert order_node_ext["W2"] == data[1]
        assert order_node_ext["W3"] == data[2]
