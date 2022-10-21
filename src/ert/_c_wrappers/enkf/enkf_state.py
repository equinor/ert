import os
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from ecl.summary import EclSum

from ert._c_wrappers.enkf.enums.ert_impl_type_enum import ErtImplType
from ert._c_wrappers.enkf.model_callbacks import LoadStatus
from ert._clib.enkf_state import get_ecl_header_file, get_ecl_unified_file

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnsembleConfig, ModelConfig, RunArg


def _internalize_GEN_DATA(
    ensemble_config: "EnsembleConfig", run_arg: "RunArg", last_report: int
):
    keys = ensemble_config.getKeylistFromImplType(ErtImplType.GEN_DATA)

    run_path = Path(run_arg.runpath)
    errors = []
    for key in keys:
        config_node = ensemble_config[key]
        filename_fmt = config_node.get_enkf_infile()
        data = []
        for i in config_node.getModelConfig().getReportSteps():
            filename = filename_fmt % i
            if not Path.exists(run_path / filename):
                errors.append(f"{key} report step {i} missing")
                continue

            with open(run_path / filename, "r") as f:
                key_data = [float(v.strip()) for v in f.readlines()]
                data.append(key_data)

            run_arg.sim_fs.save_gen_data(f"{key}-{i}", data, run_arg.iens)
        # with open(f"{run_arg.runpath}/{filename}_active", "r") as f:
        #     active_mask = [bool(v.strip()) for v in f.readlines()]
    if errors:
        return (LoadStatus.LOAD_FAILURE, "\n".join(errors))
    return (LoadStatus.LOAD_SUCCESSFUL, "")


def _internalize_SUMMARY_DATA(ens_config: "EnsembleConfig", run_arg: "RunArg"):
    summary_keys = ens_config.get_summary_keys()
    if len(summary_keys) == 0:
        return (LoadStatus.LOAD_SUCCESSFUL, "")
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
    return (LoadStatus.LOAD_SUCCESSFUL, "")


def _internalize_results(
    ens_config: "EnsembleConfig", model_config: "ModelConfig", run_arg: "RunArg"
) -> Tuple[LoadStatus, str]:

    status = _internalize_SUMMARY_DATA(ens_config, run_arg)
    if status[0] != LoadStatus.LOAD_SUCCESSFUL:
        return status

    last_report = run_arg.sim_fs.getTimeMap().last_step()
    if last_report < 0:
        last_report = model_config.get_last_history_restart()
    result = _internalize_GEN_DATA(ens_config, run_arg, last_report)

    if result[0] == LoadStatus.LOAD_FAILURE:
        return (LoadStatus.LOAD_FAILURE, "Failed to internalize GEN_DATA")
    return (LoadStatus.LOAD_SUCCESSFUL, "Results loaded successfully.")


__all__ = ["_internalize_results"]
