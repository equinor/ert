from typing_extensions import TypedDict

from ert.config import ErtConfig, Field, GenKwConfig, SurfaceConfig


class ObservationCount(TypedDict):
    observation_key: str
    count: int


class ErtSummary:
    def __init__(self, ert_config: ErtConfig) -> None:
        self.ert_config = ert_config

    def getForwardModels(self) -> list[str]:
        return self.ert_config.forward_model_step_name_list()

    def getParameters(self) -> tuple[list[str], int]:
        parameters = []
        count = 0
        for (
            key,
            config,
        ) in self.ert_config.ensemble_config.parameter_configs.items():
            match config:
                case GenKwConfig():
                    parameters.append(f"{key} ({len(config)})")
                    count += len(config)
                case Field(nx=nx, ny=ny, nz=nz):
                    parameters.append(f"{key} ({nx}, {ny}, {nz})")
                    count += len(config)
                case SurfaceConfig(ncol=ncol, nrow=nrow):
                    parameters.append(f"{key} ({ncol}, {nrow})")
                    count += len(config)
        return sorted(parameters, key=lambda k: k.lower()), count

    def getObservations(self) -> list[ObservationCount]:
        counts: list[ObservationCount] = []
        for df in self.ert_config.observations.values():
            counts.extend(df.group_by("observation_key").len(name="count").to_dicts())  # type: ignore

        return sorted(counts, key=lambda k: k["observation_key"].lower())
