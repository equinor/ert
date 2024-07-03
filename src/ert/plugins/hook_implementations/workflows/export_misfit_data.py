from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import pandas as pd

from ert.config.ert_script import ErtScript
from ert.exceptions import StorageError

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.storage import Ensemble


class ExportMisfitDataJob(ErtScript):
    """
    Will export misfit per observation and realization to a hdf file.
    The hdf file has the observation as key, and the misfit as values.
    The filename is "misfit.hdf" by default, but can be overridden by giving
    the filename as the first parameter:
    EXPORT_MISFIT_DATA path/to/output.hdf
    The misfit its calculated as follows:
    ((response_value - observation_data) / observation_std)**2
    """

    def run(
        self, ert_config: ErtConfig, ensemble: Ensemble, workflow_args: List[Any]
    ) -> None:
        target_file = "misfit.hdf" if not workflow_args else workflow_args[0]

        realizations = ensemble.get_realization_list_with_responses()

        from ert import LibresFacade  # noqa: PLC0415 (circular import)

        facade = LibresFacade(ert_config)
        misfit = facade.load_all_misfit_data(ensemble)
        if len(realizations) == 0 or misfit.empty:
            raise StorageError("No responses loaded")
        misfit.columns = pd.Index([val.split(":")[1] for val in misfit.columns])
        misfit = misfit.drop("TOTAL", axis=1)
        misfit.to_hdf(target_file, key="misfit", mode="w")
