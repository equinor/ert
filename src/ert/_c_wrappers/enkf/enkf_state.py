from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ert._c_wrappers.enkf import SummaryConfig
from ert.load_status import LoadResult, LoadStatus

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnsembleConfig, RunArg

logger = logging.getLogger(__name__)


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


__all__ = ["_write_responses_to_storage"]
