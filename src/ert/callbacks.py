from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Tuple

from ert.config import ParameterConfig, ResponseConfig, SummaryConfig
from ert.run_arg import RunArg

from .load_status import LoadResult, LoadStatus
from .realization_state import RealizationState

if TYPE_CHECKING:
    from ert.storage import EnsembleAccessor

CallbackArgs = Tuple[RunArg, Mapping[str, ResponseConfig]]
Callback = Callable[[RunArg, Mapping[str, ResponseConfig]], LoadResult]

logger = logging.getLogger(__name__)


def _read_parameters(
    runpath: str,
    iens: int,
    parameter_configurations: Iterable[ParameterConfig],
    ensemble_storage: EnsembleAccessor,
) -> LoadResult:
    result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    error_msg = ""
    for config_node in parameter_configurations:
        if not config_node.forward_init:
            continue
        try:
            start_time = time.perf_counter()
            print(f"Starting to load parameter: {config_node.name}")
            ds = config_node.read_from_runpath(Path(runpath), iens)
            ensemble_storage.save_parameters(config_node.name, iens, ds)
            print(
                f"Saved {config_node.name} to storage",
                {"Time": f"{(time.perf_counter() - start_time):.4f}s"},
            )
        except ValueError as err:
            error_msg += str(err)
            result = LoadResult(LoadStatus.LOAD_FAILURE, error_msg)
    return result


def _write_responses_to_storage(
    response_configs: Mapping[str, ResponseConfig],
    runpath: str,
    iens: int,
    ensemble_storage: EnsembleAccessor,
) -> LoadResult:
    errors = []
    for config in response_configs.values():
        if isinstance(config, SummaryConfig) and not config.keys:
            # Nothing to load, should not be handled here, should never be
            # added in the first place
            continue
        try:
            start_time = time.perf_counter()
            logger.info(f"Starting to load response: {config.name}")
            ds = config.read_from_file(runpath, iens)
            ensemble_storage.save_response(config.name, ds, iens)
            logger.info(
                f"Saved {config.name} to storage",
                extra={"Time": f"{(time.perf_counter() - start_time):.4f}s"},
            )
        except ValueError as err:
            errors.append(str(err))
    if errors:
        return LoadResult(LoadStatus.LOAD_FAILURE, "\n".join(errors))
    return LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")


def forward_model_ok(
    run_arg: RunArg,
    response_configs: Mapping[str, ResponseConfig],
    update_state_map: bool = True,
) -> LoadResult:
    parameters_result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    response_result = LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
    try:
        # We only read parameters after the prior, after that, ERT
        # handles parameters
        if run_arg.itr == 0:
            parameters_result = _read_parameters(
                run_arg.runpath,
                run_arg.iens,
                run_arg.ensemble_storage.experiment.parameter_configuration.values(),
                run_arg.ensemble_storage,
            )

        if parameters_result.status == LoadStatus.LOAD_SUCCESSFUL:
            response_result = _write_responses_to_storage(
                response_configs,
                run_arg.runpath,
                run_arg.iens,
                run_arg.ensemble_storage,
            )

    except BaseException as err:  # pylint: disable=broad-exception-caught
        logger.exception(f"Failed to load results for realization {run_arg.iens}")
        parameters_result = LoadResult(
            LoadStatus.LOAD_FAILURE,
            "Failed to load results for realization "
            f"{run_arg.iens}, failed with: {err}",
        )

    final_result = parameters_result
    if response_result.status != LoadStatus.LOAD_SUCCESSFUL:
        final_result = response_result

    if update_state_map:
        run_arg.ensemble_storage.state_map[run_arg.iens] = (
            RealizationState.HAS_DATA
            if final_result.status == LoadStatus.LOAD_SUCCESSFUL
            else RealizationState.LOAD_FAILURE
        )

    return final_result


def forward_model_exit(run_arg: RunArg, _: Mapping[str, ResponseConfig]) -> LoadResult:
    run_arg.ensemble_storage.state_map[run_arg.iens] = RealizationState.LOAD_FAILURE
    return LoadResult(None, "")
