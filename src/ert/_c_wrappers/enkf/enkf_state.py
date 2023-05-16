from __future__ import annotations

import ctypes
import logging
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ecl.summary import EclSum

from ert.load_status import LoadResult, LoadStatus

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnsembleConfig, RunArg
    from ert._c_wrappers.enkf.config.gen_data_config import GenDataConfig

logger = logging.getLogger(__name__)


def _internalize_GEN_DATA(
    ensemble_config: EnsembleConfig, run_arg: RunArg
) -> LoadResult:
    run_path = Path(run_arg.runpath)
    errors = []
    all_data = {}
    for key in ensemble_config.get_keylist_gen_data():
        config_node: GenDataConfig = ensemble_config.getNode(key)  # type: ignore
        filename_fmt = config_node.input_file
        for i in config_node.getReportSteps():
            filename = filename_fmt % i
            if not Path.exists(run_path / filename):
                errors.append(f"{key} report step {i} missing")
                continue

            all_data[f"{key}@{i}"] = np.loadtxt(run_path / filename)

    run_arg.ensemble_storage.save_gen_data(all_data, run_arg.iens)
    if errors:
        return LoadResult(LoadStatus.LOAD_FAILURE, "\n".join(errors))
    return LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")


def _should_load_summary_key(data_key, user_set_keys):
    return any(fnmatch(data_key, key) for key in user_set_keys)


def _load_summary_data(run_path: str, job_name: str) -> EclSum:
    try:
        summary = EclSum(
            f"{run_path}/{job_name}",
            include_restart=False,
            lazy_load=False,
        )
    except IOError as e:
        raise IOError("Failed to load summary data from disk.") from e

    return summary


def _internalize_SUMMARY_DATA(
    ens_config: EnsembleConfig, run_arg: RunArg, summary: EclSum
) -> LoadResult:
    data = []
    keys = []
    time_map = summary.alloc_time_vector(True)
    axis = [t.datetime() for t in time_map]

    if ens_config.refcase:
        existing_time_map = ens_config.refcase.alloc_time_vector(True)
        missing = []
        for step, (response_t, reference_t) in enumerate(
            zip(time_map, existing_time_map)
        ):
            if response_t not in existing_time_map:
                missing.append((response_t, reference_t, step + 1))
        if missing:
            logger.warning(
                f"Realization: {run_arg.iens}, load warning: {len(missing)} "
                "inconsistencies in time map, first: "
                f"Time mismatch for step: {missing[0][2]}, response time: "
                f"{missing[0][0]}, reference case: {missing[0][1]}, last: Time "
                f"mismatch for step: {missing[-1][2]}, response time: {missing[-1][0]}"
                f", reference case: {missing[-1][1]} from: {run_arg.runpath}/"
                f"{run_arg.eclbase}.UNSMRY"
            )

    user_summary_keys = ens_config.get_user_summary_keys()
    for key in summary:
        if not _should_load_summary_key(key, user_summary_keys):
            continue
        keys.append(key)

        np_vector = np.zeros(len(time_map))
        summary._init_numpy_vector_interp(
            key,
            time_map,
            np_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        data.append(np_vector)
    total = np.stack(data, axis=0)
    run_arg.ensemble_storage.save_summary_data(total, keys, axis, run_arg.iens)
    return LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")


def _write_summary_data_to_storage(
    ens_config: EnsembleConfig, run_arg: RunArg
) -> LoadResult:
    user_summary_keys = ens_config.get_user_summary_keys()
    if not user_summary_keys:
        return LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")

    try:
        summary = _load_summary_data(run_arg.runpath, run_arg.eclbase)
    except IOError:
        return LoadResult(
            LoadStatus.LOAD_FAILURE,
            "Could not find SUMMARY file or using non unified SUMMARY "
            f"file from: {run_arg.runpath}/{run_arg.eclbase}.UNSMRY",
        )

    return _internalize_SUMMARY_DATA(ens_config, run_arg, summary)


def _write_gen_data_to_storage(
    ens_config: EnsembleConfig, run_arg: RunArg
) -> LoadResult:
    result = _internalize_GEN_DATA(ens_config, run_arg)

    if result.status == LoadStatus.LOAD_FAILURE:
        return LoadResult(LoadStatus.LOAD_FAILURE, "Failed to internalize GEN_DATA")
    return LoadResult(LoadStatus.LOAD_SUCCESSFUL, "Results loaded successfully.")


__all__ = ["_write_summary_data_to_storage", "_write_gen_data_to_storage"]
