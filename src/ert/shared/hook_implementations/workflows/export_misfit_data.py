import pandas as pd

from ert._c_wrappers.enkf.enums import RealizationStateEnum
from ert._c_wrappers.job_queue import ErtScript
from ert.analysis._es_update import _get_obs_and_measure_data
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

    def run(self, target_file=None):
        ert = self.ert()

        if target_file is None:
            target_file = "misfit.hdf"
        realizations = self.ensemble.realizationList(
            RealizationStateEnum.STATE_HAS_DATA
        )

        if not realizations:
            raise StorageError("No responses loaded")

        all_observations = [(n.getObsKey(), []) for n in ert.getObservations()]
        measured_data, obs_data = _get_obs_and_measure_data(
            ert.getObservations(), self.ensemble, all_observations, realizations
        )
        joined = obs_data.join(measured_data, on=["data_key", "axis"], how="inner")
        misfit = pd.DataFrame(index=joined.index)
        for col in measured_data:
            misfit[col] = ((joined["OBS"] - joined[col]) / joined["STD"]) ** 2
        misfit.groupby("key").sum().T.to_hdf(target_file, key="misfit", mode="w")
