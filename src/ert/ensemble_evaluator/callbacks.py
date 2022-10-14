from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from ert._c_wrappers.enkf.data.enkf_node import EnkfNode
from ert._c_wrappers.enkf.enkf_state import internalize_results
from ert._c_wrappers.enkf.model_callbacks import LoadStatus
from ert._c_wrappers.enkf.node_id import NodeId
from ert._c_wrappers.enkf.state_map import RealizationStateEnum

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnsembleConfig, ModelConfig, RunArg


def _ensemble_config_forward_init(
    ens_config: "EnsembleConfig", run_arg: "RunArg"
) -> Tuple[LoadStatus, str]:
    result = LoadStatus.LOAD_SUCCESSFUL
    error_msg = ""
    iens = run_arg.iens
    for config_key in ens_config.alloc_keylist():
        config_node = ens_config[config_key]
        if not config_node.getUseForwardInit():
            continue

        node = EnkfNode(config_node)
        node_id = NodeId(report_step=0, realization_number=run_arg.iens)

        if node.has_data(run_arg.sim_fs, node_id):
            # Already initialised, ignore
            continue

        if node.forward_init(run_arg.runpath, iens):
            node.save(run_arg.sim_fs, node_id)
        else:
            init_file = Path(config_node.get_init_file_fmt() % (iens,))
            if not init_file.exists():
                error_msg = (
                    "Failed to initialize node "
                    f"'{node.name()}' in file {init_file}: File not found\n"
                )
            else:
                error_msg = (
                    f"Failed to initialize node '{node.name()}' in file {init_file}\n"
                )

            result = LoadStatus.LOAD_FAILURE
    return (result, error_msg)


def forward_model_ok(
    run_arg: "RunArg",
    ens_conf: "EnsembleConfig",
    model_conf: "ModelConfig",
) -> Tuple[LoadStatus, str]:
    result = (LoadStatus.LOAD_SUCCESSFUL, "")
    if ens_conf.have_forward_init():
        result = _ensemble_config_forward_init(ens_conf, run_arg)

    if result[0] == LoadStatus.LOAD_SUCCESSFUL:
        result = internalize_results(ens_conf, model_conf, run_arg)

    run_arg.sim_fs.getStateMap()[run_arg.iens] = (
        RealizationStateEnum.STATE_HAS_DATA
        if result[0] == LoadStatus.LOAD_SUCCESSFUL
        else RealizationStateEnum.STATE_LOAD_FAILURE
    )

    return result
