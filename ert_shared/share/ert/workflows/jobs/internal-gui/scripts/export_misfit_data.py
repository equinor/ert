from collections import defaultdict

import pandas as pd

from ert.exceptions import StorageError
from res.enkf import ErtScript, RealizationStateEnum

class ExportMisfitDataJob(ErtScript):
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
