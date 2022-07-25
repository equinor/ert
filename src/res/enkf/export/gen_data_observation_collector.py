from res.enkf import EnKFMain
from res.enkf.enums import EnkfObservationImplementationType
from typing import List


class GenDataObservationCollector:
    @staticmethod
    def getAllObservationKeys(ert: EnKFMain) -> List[str]:
        enkf_obs = ert.getObservations()
        observation_keys = enkf_obs.getTypedKeylist(
            EnkfObservationImplementationType.GEN_OBS
        )
        return [key for key in observation_keys]

    @staticmethod
    def getObservationKeyForDataKey(ert: EnKFMain, data_key, data_report_step) -> str:
        observation_key = None

        enkf_obs = ert.getObservations()
        for obs_vector in enkf_obs:
            if EnkfObservationImplementationType.GEN_OBS:
                report_step = obs_vector.firstActiveStep()
                key = obs_vector.getDataKey()

                if key == data_key and report_step == data_report_step:
                    observation_key = obs_vector.getObservationKey()

        return observation_key
