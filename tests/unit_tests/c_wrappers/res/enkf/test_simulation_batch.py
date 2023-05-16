import pytest

from ert._c_wrappers.enkf import EnkfConfigNode, EnKFMain
from ert._c_wrappers.enkf.config.gen_data_config import GenDataConfig
from ert._c_wrappers.enkf.enums import EnkfVarType, RealizationStateEnum
from ert.simulator.simulation_context import _run_forward_model


@pytest.mark.integration_test
def test_run_simulation_batch(setup_case, prior_ensemble):
    ert = EnKFMain(setup_case("config/simulation_batch", "config.ert"))
    ens_size = 4
    ens_config = ert.ensembleConfig()

    # Observe that a significant amount of hardcoding
    # regarding the GEN_DATA and EXT_PARAM nodes is assumed
    # between this test, the config file and the forward model.

    # Add control nodes
    order_control = EnkfConfigNode.create_ext_param("WELL_ORDER", ["W1", "W2", "W3"])
    ens_config.add_config_node_meta(
        key="WELL_ORDER",
        output_file="WELL_ORDER.json",
        var_type=EnkfVarType.EXT_PARAMETER,
    )
    injection_control = EnkfConfigNode.create_ext_param("WELL_INJECTION", ["W1", "W4"])
    ens_config.add_config_node_meta(
        key="WELL_INJECTION",
        output_file="WELL_INJECTION.json",
        var_type=EnkfVarType.EXT_PARAMETER,
    )
    ens_config.addNode(order_control)
    ens_config.addNode(injection_control)

    # Add result nodes
    order_result = GenDataConfig("ORDER")
    injection_result = GenDataConfig("INJECTION")
    ens_config.add_config_node_meta(
        key="ORDER",
        input_file="order_%d",
        var_type=EnkfVarType.DYNAMIC_RESULT,
    )
    ens_config.add_config_node_meta(
        key="INJECTION",
        input_file="injection_%d",
        var_type=EnkfVarType.DYNAMIC_RESULT,
    )
    ens_config.addNode(order_result)
    ens_config.addNode(injection_result)

    batch_size = ens_size
    order_node_ext = {}
    injection_node_ext = {}
    for iens in range(batch_size):
        order_node_ext["W1"] = iens
        order_node_ext["W2"] = iens * 10
        order_node_ext["W3"] = iens * 100
        prior_ensemble.save_ext_param("WELL_ORDER", iens, order_node_ext)

        injection_node_ext["W1"] = iens + 1
        injection_node_ext["W4"] = 3 * (iens + 1)
        prior_ensemble.save_ext_param("WELL_INJECTION", iens, injection_node_ext)
        prior_ensemble.state_map[iens] = RealizationStateEnum.STATE_INITIALIZED

    mask = [True] * batch_size
    run_context = ert.ensemble_context(prior_ensemble, mask, iteration=0)
    ert.createRunPath(run_context)
    job_queue = ert.get_queue_config().create_job_queue()

    ert.createRunPath(run_context)
    num = _run_forward_model(ert, job_queue, run_context)
    assert num == batch_size

    for iens in range(batch_size):
        data, _ = prior_ensemble.load_gen_data("ORDER@0", [iens])
        data = data.flatten()

        order_node_ext = prior_ensemble.load_ext_param("WELL_ORDER", iens)
        assert order_node_ext["W1"] == data[0]
        assert order_node_ext["W2"] == data[1]
        assert order_node_ext["W3"] == data[2]
