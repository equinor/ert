import ctypes
import logging
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np
from ecl.summary import EclSum

from ert._c_wrappers.enkf.enums.ert_impl_type_enum import ErtImplType
from ert._c_wrappers.enkf.model_callbacks import LoadStatus

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnsembleConfig, RunArg

logger = logging.getLogger(__name__)


def _internalize_GEN_DATA(ensemble_config: "EnsembleConfig", run_arg: "RunArg"):
    keys = ensemble_config.getKeylistFromImplType(ErtImplType.GEN_DATA)

    run_path = Path(run_arg.runpath)
    errors = []
    all_data = {}
    for key in keys:
        config_node = ensemble_config[key]
        filename_fmt = ensemble_config.get_enkf_infile(key)
        for i in config_node.getModelConfig().getReportSteps():
            filename = filename_fmt % i
            if not Path.exists(run_path / filename):
                errors.append(f"{key} report step {i} missing")
                continue

            with open(run_path / filename, "r", encoding="utf-8") as f:
                data = [float(v.strip()) for v in f.readlines()]
            all_data[f"{key}@{i}"] = np.array(data)

    run_arg.ensemble_storage.save_gen_data(all_data, run_arg.iens)
    if errors:
        return (LoadStatus.LOAD_FAILURE, "\n".join(errors))
    return (LoadStatus.LOAD_SUCCESSFUL, "")


def _should_load_summary_key(data_key, user_set_keys):
    return any(fnmatch(data_key, key) for key in user_set_keys)


def _internalize_SUMMARY_DATA(ens_config: "EnsembleConfig", run_arg: "RunArg"):
    user_summary_keys = ens_config.get_summary_keys()
    if len(user_summary_keys) == 0:
        return (LoadStatus.LOAD_SUCCESSFUL, "")

    try:
        summary = EclSum(
            f"{run_arg.runpath}/{run_arg.job_name}",
            include_restart=False,
            lazy_load=False,
        )
    except IOError:
        return (
            LoadStatus.LOAD_FAILURE,
            "Could not find SUMMARY file or using non unified SUMMARY "
            f"file from: {run_arg.runpath}/{run_arg.job_name}.UNSMRY",
        )
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
                f"{run_arg.job_name}.UNSMRY"
            )

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
    return (LoadStatus.LOAD_SUCCESSFUL, "")


def _internalize_results(
    ens_config: "EnsembleConfig", run_arg: "RunArg"
) -> Tuple[LoadStatus, str]:
    status = _internalize_SUMMARY_DATA(ens_config, run_arg)
    if status[0] != LoadStatus.LOAD_SUCCESSFUL:
        return status

    result = _internalize_GEN_DATA(ens_config, run_arg)

    if result[0] == LoadStatus.LOAD_FAILURE:
        return (LoadStatus.LOAD_FAILURE, "Failed to internalize GEN_DATA")
    return (LoadStatus.LOAD_SUCCESSFUL, "Results loaded successfully.")


__all__ = ["_internalize_results"]
