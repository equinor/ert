from typing import Optional

from ert import ErtScript
from ert.exceptions import StorageError


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

    def run(self, ert_config, ensemble, target_file: Optional[str] = None) -> None:
        if target_file is None:
            target_file = "misfit.hdf"

        realizations = ensemble.get_realization_list_with_responses()

        if not realizations:
            raise StorageError("No responses loaded")
        from ert import LibresFacade

        facade = LibresFacade(ert_config)
        misfit = facade.load_all_misfit_data(ensemble)
        misfit.columns = [val.split(":")[1] for val in misfit.columns]
        misfit = misfit.drop("TOTAL", axis=1)
        misfit.to_hdf(target_file, key="misfit", mode="w")
