import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

from ert._c_wrappers.enkf.data.enkf_node import EnkfNode
from ert._c_wrappers.enkf.enkf_state import _internalize_results
from ert._c_wrappers.enkf.enums import ErtImplType
from ert._c_wrappers.enkf.model_callbacks import LoadStatus
from ert._c_wrappers.enkf.node_id import NodeId
from ert._c_wrappers.enkf.state_map import RealizationStateEnum

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnsembleConfig, RunArg

logger = logging.getLogger(__name__)


def forward_model_ok(
    run_arg: "RunArg",
    ens_conf: "EnsembleConfig",
    num_steps: int,
) -> Tuple[LoadStatus, str]:
    try:
        result = (LoadStatus.LOAD_SUCCESSFUL, "")
        if ens_conf.have_forward_init():
            forward_init_config_nodes = ens_conf.check_forward_init_nodes()
            error_msg = ""
            for config_node in forward_init_config_nodes:
                if not config_node.getUseForwardInit():
                    continue
                if config_node.getImplementationType() == ErtImplType.SURFACE:
                    run_path = Path(run_arg.runpath)
                    file_name = config_node.get_init_file_fmt()
                    if "%d" in file_name:
                        file_name = file_name % run_arg.iens
                    file_path = run_path / file_name
                    if file_path.exists():
                        run_arg.ensemble_storage.save_surface_file(
                            config_node.getKey(), run_arg.iens, file_path
                        )
                    else:
                        error_msg += (
                            "Failed to initialize parameter "
                            f"'{config_node.getKey()}' in file {file_name}: "
                            "File not found\n"
                        )
                        result = (LoadStatus.LOAD_FAILURE, error_msg)

                    continue

                node = EnkfNode(config_node)
                node_id = NodeId(report_step=0, iens=run_arg.iens)

                if node.has_data(run_arg.ensemble_storage, node_id):
                    # Already initialised, ignore
                    continue

                if node.forward_init(run_arg.runpath, run_arg.iens):
                    node.save(run_arg.ensemble_storage, node_id)
                else:
                    if "%d" in config_node.get_init_file_fmt():
                        init_file = Path(config_node.get_init_file_fmt() % (run_arg.iens,))
                    else:
                        init_file = Path(config_node.get_init_file_fmt())
                    if not init_file.exists():
                        error_msg += (
                            "Failed to initialize node "
                            f"'{node.name()}' in file {init_file}: File not found\n"
                        )
                    else:
                        error_msg += (
                            f"Failed to initialize node '{node.name()}' "
                            "in file {init_file}\n"
                        )

                    result = (LoadStatus.LOAD_FAILURE, error_msg)

        if result[0] == LoadStatus.LOAD_SUCCESSFUL:
            result = _internalize_results(ens_conf, num_steps, run_arg)

    except Exception:
        logging.exception("Unhandled exception in callback for forward_model")
        result = (
            LoadStatus.LOAD_FAILURE,
            "Unhandled exception in callback for forward_model",
        )

    state_map = run_arg.ensemble_storage.getStateMap()
    state_map._set(
        run_arg.iens,
        RealizationStateEnum.STATE_HAS_DATA
        if result[0] == LoadStatus.LOAD_SUCCESSFUL
        else RealizationStateEnum.STATE_LOAD_FAILURE,
    )

    return result


def forward_model_exit(run_arg: "RunArg", *_: Tuple[Any]):
    run_arg.ensemble_storage.getStateMap()[
        run_arg.iens
    ] = RealizationStateEnum.STATE_LOAD_FAILURE
