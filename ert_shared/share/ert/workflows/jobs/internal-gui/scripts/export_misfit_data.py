import pandas as pd
from res.enkf import ErtScript, RealizationStateEnum


class ExportMisfitDataJob(ErtScript):
    def run(self, target_file=None):
        ert = self.ert()
        fs = ert.getEnkfFsManager().getCurrentFileSystem()

        if target_file is None:
            target_file = "misfit.csv"

        pd.DataFrame(
            {
                "MISFIT%s"
                % obs_vector.getObservationKey(): obs_vector.getTotalCh2(
                    fs, realization_nr
                )
                for obs_vector in ert.getObservations()
                for realization_nr, has_data in enumerate(
                    fs.getStateMap().selectMatching(RealizationStateEnum.STATE_HAS_DATA)
                )
                if has_data
            }
        ).to_csv(target_file)
