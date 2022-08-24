from collections import defaultdict

import pandas as pd

from ert._c_wrappers.enkf import RealizationStateEnum
from ert._c_wrappers.job_queue import ErtScript
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
        fs = ert.getEnkfFsManager().getCurrentFileSystem()

        if target_file is None:
            target_file = "misfit.hdf"
        realizations = [
            i
            for i, has_data in enumerate(
                fs.getStateMap().selectMatching(RealizationStateEnum.STATE_HAS_DATA)
            )
            if has_data
        ]
        if not realizations:
            raise StorageError("No responses loaded")
        misfits = defaultdict(list)
        for realization in realizations:
            for obs_vector in ert.getObservations():
                misfits[obs_vector.getObservationKey()].append(
                    obs_vector.getTotalChi2(fs, realization)
                )
        pd.DataFrame(misfits, index=realizations).to_hdf(
            target_file, key="misfit", mode="w"
        )
