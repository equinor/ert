from typing import List, Tuple

from ert.config import (
    EnkfObservationImplementationType,
    Field,
    GenKwConfig,
    SurfaceConfig,
)
from ert.enkf_main import EnKFMain


class ErtSummary:
    def __init__(self, ert: EnKFMain):
        self.ert = ert

    def getForwardModels(self) -> List[str]:
        return self.ert.resConfig().forward_model_job_name_list()

    def getParameters(self) -> Tuple[List[str], int]:
        parameters = []
        count = 0
        for key, config in self.ert.ensembleConfig().parameter_configs.items():
            if isinstance(config, GenKwConfig):
                parameters.append(f"{key} ({len(config.transfer_functions)})")
                count += len(config.transfer_functions)
            if isinstance(config, Field):
                parameters.append(f"{key} ({config.nx}, {config.ny}, {config.nz})")
                count += config.nx * config.ny * config.nz
            if isinstance(config, SurfaceConfig):
                parameters.append(f"{key} ({config.ncol}, {config.nrow})")
                count += config.ncol * config.nrow
        return sorted(parameters, key=lambda k: k.lower()), count

    def getObservations(self) -> List[str]:
        gen_obs = self.ert.getObservations().getTypedKeylist(
            EnkfObservationImplementationType.GEN_OBS
        )

        summary_obs = self.ert.getObservations().getTypedKeylist(
            EnkfObservationImplementationType.SUMMARY_OBS
        )

        keys = []
        summary_keys_count = {}
        summary_keys = []
        for key in summary_obs:
            data_key = self.ert.getObservations()[key].data_key

            if data_key not in summary_keys_count:
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
