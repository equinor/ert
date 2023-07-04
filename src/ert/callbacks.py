from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Tuple

from ert._c_wrappers.enkf import EnsembleConfig, RunArg, SummaryConfig
from ert._c_wrappers.enkf.config.parameter_config import ParameterConfig
from ert._c_wrappers.enkf.enums import RealizationStateEnum

from .load_status import LoadResult, LoadStatus

CallbackArgs = Tuple[RunArg, EnsembleConfig]
Callback = Callable[[RunArg, EnsembleConfig], LoadResult]

logger = logging.getLogger(__name__)


def _read_parameters(
    run_arg: RunArg, parameter_configuration: Iterable[ParameterConfig]
) -> LoadResult:
    result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    error_msg = ""
    for config_node in parameter_configuration:
        if not config_node.forward_init:
            continue
        try:
            ds = config_node.read_from_runpath(Path(run_arg.runpath), run_arg.iens)
            run_arg.ensemble_storage.save_parameters(config_node.name, run_arg.iens, ds)
        except ValueError as err:
            error_msg += str(err)
            result = LoadResult(LoadStatus.LOAD_FAILURE, error_msg)
    return result


def _write_responses_to_storage(
    ens_config: EnsembleConfig, run_arg: RunArg
) -> LoadResult:
    errors = []
    for config in ens_config.response_configs.values():
        if isinstance(config, SummaryConfig):
            # Nothing to load, should not be handled here, should never be
            # added in the first place
            if not config.keys:
                continue
        try:
            ds = config.read_from_file(run_arg.runpath, run_arg.iens)
            run_arg.ensemble_storage.save_response(config.name, ds, run_arg.iens)
        except ValueError as err:
            errors.append(str(err))
    if errors:
        return LoadResult(LoadStatus.LOAD_FAILURE, "\n".join(errors))
    return LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")


def forward_model_ok(
    run_arg: RunArg,
    ens_conf: EnsembleConfig,
) -> LoadResult:
    parameters_result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    response_result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    try:
        # We only read parameters after the prior, after that, ERT
        # handles parameters
        if run_arg.itr == 0:
            parameters_result = _read_parameters(
                run_arg,
                run_arg.ensemble_storage.experiment.parameter_configuration.values(),
            )

        if parameters_result.status == LoadStatus.LOAD_SUCCESSFUL:
            response_result = _write_responses_to_storage(ens_conf, run_arg)

    except Exception as err:
        logging.exception("Unhandled exception in callback for forward_model")
        parameters_result = LoadResult(
            LoadStatus.LOAD_FAILURE,
            f"Unhandled exception in callback for forward_model {err}",
        )

    final_result = parameters_result
    if response_result.status != LoadStatus.LOAD_SUCCESSFUL:
        final_result = response_result

    run_arg.ensemble_storage.state_map[run_arg.iens] = (
        RealizationStateEnum.STATE_HAS_DATA
        if final_result.status == LoadStatus.LOAD_SUCCESSFUL
        else RealizationStateEnum.STATE_LOAD_FAILURE
    )

    return final_result


def forward_model_exit(run_arg: RunArg, _: EnsembleConfig) -> LoadResult:
    run_arg.ensemble_storage.state_map[
        run_arg.iens
    ] = RealizationStateEnum.STATE_LOAD_FAILURE
    return LoadResult(None, "")
