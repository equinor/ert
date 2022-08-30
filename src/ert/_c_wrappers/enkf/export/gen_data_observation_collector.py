from typing import List, Optional

from ert._c_wrappers.enkf import EnKFMain
from ert._c_wrappers.enkf.enums import EnkfObservationImplementationType


class GenDataObservationCollector:
    @staticmethod
    def getAllObservationKeys(ert: EnKFMain) -> List[str]:
        return list(
            ert.getObservations().getTypedKeylist(
                EnkfObservationImplementationType.GEN_OBS
            )
        )

    @staticmethod
    def getObservationKeyForDataKey(
        ert: EnKFMain, data_key, data_report_step
    ) -> Optional[str]:
        observation_key = None

        enkf_obs = ert.getObservations()
        for obs_vector in enkf_obs:
            if EnkfObservationImplementationType.GEN_OBS:
                report_step = obs_vector.firstActiveStep()
                key = obs_vector.getDataKey()

                if key == data_key and report_step == data_report_step:
                    observation_key = obs_vector.getObservationKey()

        return observation_key
