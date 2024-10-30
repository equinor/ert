from typing import List, Tuple, TypedDict

from ert.config import ErtConfig, Field, GenKwConfig, SurfaceConfig


class ObservationCount(TypedDict):
    observation_key: str
    count: int


class ErtSummary:
    def __init__(self, ert_config: ErtConfig):
        self.ert_config = ert_config

    def getForwardModels(self) -> List[str]:
        return self.ert_config.forward_model_step_name_list()

    def getParameters(self) -> Tuple[List[str], int]:
        parameters = []
        count = 0
        for (
            key,
            config,
        ) in self.ert_config.ensemble_config.parameter_configs.items():
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

    def getObservations(self) -> List[ObservationCount]:
        counts: List[ObservationCount] = []
        for df in self.ert_config.observations.values():
            counts.extend(df.group_by("observation_key").count().to_dicts())  #  type: ignore

        return sorted(counts, key=lambda k: k["observation_key"].lower())
