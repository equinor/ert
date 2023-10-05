from typing import List, Set, Tuple

from ert.config import (
    EnkfObservationImplementationType,
    Field,
    GenKwConfig,
    SurfaceConfig,
)
from ert.config.summary_observation import SummaryObservation
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
        gen_obs = self.ert.getObservations().getTypedKeylist(
            EnkfObservationImplementationType.GEN_OBS
        )

        summary_obs = self.ert.getObservations().getTypedKeylist(
            EnkfObservationImplementationType.SUMMARY_OBS
        )

        summary_keys: Set[str] = set()
        for key in summary_obs:
            for obs in self.ert.getObservations()[key].observations.values():
                if isinstance(obs, SummaryObservation):
                    summary_keys.add(obs.observation_key)

        obs_keys = set(gen_obs) | summary_keys
        return sorted(obs_keys, key=lambda k: k.lower())
