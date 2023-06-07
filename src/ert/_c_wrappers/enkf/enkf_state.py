from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ert._c_wrappers.enkf import SummaryConfig
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
    for key in ensemble_config.get_keylist_gen_data():
        datasets = []
        config_node: GenDataConfig = ensemble_config.response_configs[key]
        filename_fmt = config_node.input_file
        for report_step in config_node.getReportSteps():
            filename = filename_fmt % report_step
            if not Path.exists(run_path / filename):
                errors.append(f"{key} report step {report_step} missing")
                continue

            data = np.loadtxt(run_path / filename, ndmin=1)
            active_information_file = run_path / (filename + "_active")
            if active_information_file.exists():
                index_list = np.flatnonzero(np.loadtxt(active_information_file))
                data = data[index_list]
            else:
                index_list = np.arange(len(data))
            datasets.append(
                xr.Dataset(
                    {"values": (["report_step", "index"], [data])},
                    coords={"index": index_list, "report_step": [report_step]},
                )
            )
        run_arg.ensemble_storage.save_response(
            key, xr.combine_by_coords(datasets), run_arg.iens
        )
    if errors:
        return LoadResult(LoadStatus.LOAD_FAILURE, "\n".join(errors))
    return LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")


def _write_summary_data_to_storage(
    ens_config: EnsembleConfig, run_arg: RunArg
) -> LoadResult:
    for config in ens_config.response_configs.values():
        if isinstance(config, SummaryConfig):
            # Nothing to load, should not be handled here, should never be
            # added in the first place
            if not config.keys:
                return LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")
            try:
                ds = config.read_from_file(run_arg.runpath, run_arg.iens)
                run_arg.ensemble_storage.save_response("summary", ds, run_arg.iens)
            except ValueError as err:
                return LoadResult(LoadStatus.LOAD_FAILURE, str(err))
    return LoadResult(LoadStatus.LOAD_SUCCESSFUL, "")


def _write_gen_data_to_storage(
    ens_config: EnsembleConfig, run_arg: RunArg
) -> LoadResult:
    result = _internalize_GEN_DATA(ens_config, run_arg)

    if result.status == LoadStatus.LOAD_FAILURE:
        return LoadResult(LoadStatus.LOAD_FAILURE, "Failed to internalize GEN_DATA")
    return LoadResult(LoadStatus.LOAD_SUCCESSFUL, "Results loaded successfully.")


__all__ = ["_write_summary_data_to_storage", "_write_gen_data_to_storage"]
