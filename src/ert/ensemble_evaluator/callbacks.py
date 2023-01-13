from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import xtgeo

from ert._c_wrappers.enkf.enkf_state import _internalize_results
from ert._c_wrappers.enkf.enums import ErtImplType, RealizationStateEnum
from ert._c_wrappers.enkf.model_callbacks import LoadStatus

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnsembleConfig, RunArg


def forward_model_ok(
    run_arg: "RunArg",
    ens_conf: "EnsembleConfig",
    num_steps: int,
) -> Tuple[LoadStatus, str]:
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

            if config_node.getImplementationType() == ErtImplType.FIELD:
                run_path = Path(run_arg.runpath)
                file_name = config_node.get_init_file_fmt()
                if "%d" in file_name:
                    file_name = file_name % run_arg.iens
                file_path = run_path / file_name

                if run_arg.ensemble_storage.field_has_data(
                    config_node.getKey(), run_arg.iens
                ):
                    # Already initialised, ignore
                    continue

                grid = xtgeo.grid_from_file(ens_conf.grid_file)
                props = xtgeo.gridproperty_from_file(
                    pfile=file_path, name=config_node.getKey(), grid=grid
                )

                data = props.values1d.data
                field_config = config_node.getFieldModelConfig()
                trans = field_config.get_init_transform_name()
                data_transformed = field_config.transform(trans, data)

                run_arg.ensemble_storage.save_field_data(
                    config_node.getKey(), run_arg.iens, data_transformed
                )

                continue

            raise NotImplementedError(
                f"{config_node.getImplementationType()} is not supported"
            )

    if result[0] == LoadStatus.LOAD_SUCCESSFUL:
        result = _internalize_results(ens_conf, num_steps, run_arg)

    run_arg.ensemble_storage.state_map[run_arg.iens] = (
        RealizationStateEnum.STATE_HAS_DATA
        if result[0] == LoadStatus.LOAD_SUCCESSFUL
        else RealizationStateEnum.STATE_LOAD_FAILURE
    )

    return result


def forward_model_exit(run_arg: "RunArg", *_: Tuple[Any]):
    run_arg.ensemble_storage.state_map[
        run_arg.iens
    ] = RealizationStateEnum.STATE_LOAD_FAILURE
