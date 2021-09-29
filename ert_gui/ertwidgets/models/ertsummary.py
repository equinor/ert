from typing import List

from res.enkf.enums.enkf_obs_impl_type_enum import EnkfObservationImplementationType
from res.enkf.enums.enkf_var_type_enum import EnkfVarType
from ert_shared import ERT


class ErtSummary:
    def getForwardModels(self) -> List[str]:
        forward_model = ERT.ert.getModelConfig().getForwardModel()
        return list(forward_model.joblist())

    def getParameters(self) -> List[str]:
        parameters = ERT.ert.ensembleConfig().getKeylistFromVarType(
            EnkfVarType.PARAMETER
        )
        return sorted(list(parameters), key=lambda k: k.lower())

    def getObservations(self) -> List[str]:
        gen_obs = ERT.ert.getObservations().getTypedKeylist(
            EnkfObservationImplementationType.GEN_OBS
        )

        summary_obs = ERT.ert.getObservations().getTypedKeylist(
            EnkfObservationImplementationType.SUMMARY_OBS
        )

        keys = []
        summary_keys_count = {}
        summary_keys = []
        for key in summary_obs:
            data_key = ERT.ert.getObservations()[key].getDataKey()

            if not data_key in summary_keys_count:
                summary_keys_count[data_key] = 1
                summary_keys.append(data_key)
            else:
                summary_keys_count[data_key] += 1

            if key == data_key:
                keys.append(key)
            else:
                keys.append(f"{key} [{data_key}]")

        obs_keys = list(gen_obs) + summary_keys
        return sorted(obs_keys, key=lambda k: k.lower())
