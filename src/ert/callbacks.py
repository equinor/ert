from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

from ert._c_wrappers.enkf.config.parameter_config import ParameterConfig
from ert._c_wrappers.enkf.enkf_state import (
    _write_gen_data_to_storage,
    _write_summary_data_to_storage,
)
from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert._c_wrappers.enkf.model_callbacks import LoadStatus
from ert.shared.status import LoadResult

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnsembleConfig, RunArg
    from ert._c_wrappers.enkf.ensemble_config import ParameterConfiguration

logger = logging.getLogger(__name__)


def _read_parameters(
    run_arg: RunArg, parameter_configuration: ParameterConfiguration
) -> LoadResult:
    result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    error_msg = ""
    for config_node in parameter_configuration:
        if not config_node.forward_init:
            continue
        if isinstance(config_node, ParameterConfig):
            try:
                config_node.load(
                    Path(run_arg.runpath), run_arg.iens, run_arg.ensemble_storage
                )
            except ValueError as err:
                error_msg += str(err)
                result = LoadResult(LoadStatus.LOAD_FAILURE, error_msg)
    return result


def forward_model_ok(
    run_arg: RunArg,
    ens_conf: EnsembleConfig,
) -> LoadResult:
    parameters_result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    summary_result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    gen_data_result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    try:
        # We only read parameters after the prior, after that, ERT
        # handles parameters
        if run_arg.itr == 0:
            parameters_result = _read_parameters(
                run_arg, ens_conf.parameter_configuration
            )

        if parameters_result.status == LoadStatus.LOAD_SUCCESSFUL:
            summary_result = _write_summary_data_to_storage(ens_conf, run_arg)
            gen_data_result = _write_gen_data_to_storage(ens_conf, run_arg)

    except Exception as err:
        logging.exception("Unhandled exception in callback for forward_model")
        parameters_result = LoadResult(
            LoadStatus.LOAD_FAILURE,
            f"Unhandled exception in callback for forward_model {err}",
        )

    final_result = parameters_result
    if summary_result.status != LoadStatus.LOAD_SUCCESSFUL:
        final_result = summary_result
    elif gen_data_result.status != LoadStatus.LOAD_SUCCESSFUL:
        final_result = gen_data_result

    run_arg.ensemble_storage.state_map[run_arg.iens] = (
        RealizationStateEnum.STATE_HAS_DATA
        if final_result.status == LoadStatus.LOAD_SUCCESSFUL
        else RealizationStateEnum.STATE_LOAD_FAILURE
    )

    return final_result


def forward_model_exit(run_arg: RunArg, *_: Tuple[Any]) -> LoadResult:
    run_arg.ensemble_storage.state_map[
        run_arg.iens
    ] = RealizationStateEnum.STATE_LOAD_FAILURE
    return LoadResult(None, "")
