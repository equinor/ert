from collections import Counter, defaultdict

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

    def get_parameters(self) -> tuple[list[str], int]:
        parameters = []
        genkw_groups: dict[str, int] = defaultdict(int)
        count = 0
        for config in self.ert_config.parameter_configurations_with_design_matrix:
            match config:
                case GenKwConfig(name=key):
                    genkw_groups[config.group_name] += 1
                    count += 1
                case Field(name=key, nx=nx, ny=ny, nz=nz):
                    parameters.append(f"{key} ({nx}, {ny}, {nz})")
                    count += len(config)
                case SurfaceConfig(name=key, ncol=ncol, nrow=nrow):
                    parameters.append(f"{key} ({ncol}, {nrow})")
                    count += len(config)
        parameters += [
            f"{group_name} ({cnt})" for group_name, cnt in genkw_groups.items()
        ]
        return sorted(parameters, key=lambda k: k.lower()), count

    def getObservations(self) -> list[ObservationCount]:
        name_to_types = defaultdict(set)
        for obs in self.ert_config.observation_declarations:
            name_to_types[obs.name].add(obs.type)

        multi_type_names = {
            name for name, types in name_to_types.items() if len(types) > 1
        }

        keys = (
            f"{obs.name}[{obs.type}]" if obs.name in multi_type_names else obs.name
            for obs in self.ert_config.observation_declarations
        )
        counts = Counter(keys)

        counts_list = [
            ObservationCount({"observation_key": key, "count": count})
            for key, count in counts.items()
        ]
        return sorted(counts_list, key=lambda k: k["observation_key"].lower())
