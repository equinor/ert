from typing import List, Tuple

from ert.config import Field, GenKwConfig, SurfaceConfig
from ert.enkf_main import EnKFMain


class ErtSummary:
    def __init__(self, ert: EnKFMain):
        self.ert = ert

    def getForwardModels(self) -> List[str]:
        return self.ert.ert_config.forward_model_job_name_list()

    def getParameters(self) -> Tuple[List[str], int]:
        parameters = []
        count = 0
        for (
            key,
            config,
        ) in self.ert.ert_config.ensemble_config.parameter_configs.items():
            if isinstance(config, GenKwConfig):
                parameters.append(f"{key} ({len(config)})")
                count += len(config)
            if isinstance(config, Field):
                parameters.append(f"{key} ({config.nx}, {config.ny}, {config.nz})")
                count += len(config)
            if isinstance(config, SurfaceConfig):
                parameters.append(f"{key} ({config.ncol}, {config.nrow})")
                count += len(config)
        return sorted(parameters, key=lambda k: k.lower()), count

    def getObservations(self) -> List[str]:
        obs_keys = self.ert.ert_config.observations.keys()
        return sorted(obs_keys, key=lambda k: k.lower())
