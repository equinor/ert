import ctypes
import os
from typing import TYPE_CHECKING, Tuple

import ecl_data_io
import numpy
from ecl.summary import EclSum

from ert._c_wrappers.enkf.model_callbacks import LoadStatus
from ert._clib.enkf_state import (
    get_ecl_header_file,
    get_ecl_unified_file,
    internalize_dynamic_eclipse_results,
    internalize_GEN_DATA,
)

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnsembleConfig, ModelConfig, RunArg


def _internalize_results(
    ens_config: "EnsembleConfig", model_config: "ModelConfig", run_arg: "RunArg"
) -> Tuple[LoadStatus, str]:
    summary_keys = ens_config.get_summary_keys()

    if len(summary_keys) > 0:
        # We are expecting there to be summary data
        # The timing information - i.e. mainly what is the last report step
        # in these results are inferred from the loading of summary results,
        # hence we must load the summary results first.

        ecl_header_file = get_ecl_header_file(run_arg.runpath, run_arg.job_name)
        ecl_unified_file = get_ecl_unified_file(run_arg.runpath, run_arg.job_name)
        if not os.path.exists(ecl_header_file) or not os.path.exists(ecl_unified_file):
            return (
                LoadStatus.LOAD_FAILURE,
                "Could not find SUMMARY file or using non unified SUMMARY "
                f"file from: {run_arg.runpath}/{run_arg.job_name}.UNSMRY",
            )

        summary = EclSum(
            f"{run_arg.runpath}/{run_arg.job_name}",
            include_restart=False,
            lazy_load=False,
        )
        run_arg.sim_fs.save_summary_data(summary, run_arg.iens)
        # summary = ecl_data_io.read(ecl_unified_file)
        # status = internalize_dynamic_eclipse_results(
        #     ens_config,
        #     summary,
        #     summary_keys,
        #     run_arg.sim_fs,
        #     run_arg.iens,
        # )
        status = (LoadStatus.LOAD_SUCCESSFUL, "")
        if status[0] != LoadStatus.LOAD_SUCCESSFUL:
            return (
                status[0],
                status[1] + f" from: {run_arg.runpath}/{run_arg.job_name}.UNSMRY",
            )

    last_report = run_arg.sim_fs.getTimeMap().last_step()
    if last_report < 0:
        last_report = model_config.get_last_history_restart()
    result = internalize_GEN_DATA(ens_config, run_arg, model_config, last_report)
    if result == LoadStatus.LOAD_FAILURE:
        return (LoadStatus.LOAD_FAILURE, "Failed to internalize GEN_DATA")
    return (LoadStatus.LOAD_SUCCESSFUL, "Results loaded successfully.")


__all__ = ["_internalize_results"]
