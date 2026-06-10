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

    def get_parameters(self) -> tuple[list[str], list[str], int]:
        parameters_updatable = []
        parameters_not_updatable = []
        genkw_groups_updadatable: dict[str, int] = defaultdict(int)
        genkw_groups_not_updadatable: dict[str, int] = defaultdict(int)
        count = 0
        for config in self.ert_config.parameter_configurations_with_design_matrix:
            match config:
                case GenKwConfig(name=key):
                    if config.update_strategy is not None:
                        if config.group_name:
                            genkw_groups_updadatable[config.group_name] += 1
                        else:
                            genkw_groups_updadatable["gen_kw"] += 1
                    elif config.group_name:
                        genkw_groups_not_updadatable[config.group_name] += 1
                    else:
                        genkw_groups_not_updadatable["gen_kw"] += 1
                    count += 1
                case Field(name=key, nx=nx, ny=ny, nz=nz):
                    if config.update_strategy is not None:
                        parameters_updatable.append(f"{key} ({nx}, {ny}, {nz})")
                    else:
                        parameters_not_updatable.append(f"{key} ({nx}, {ny}, {nz})")
                    count += len(config)
                case SurfaceConfig(name=key, ncol=ncol, nrow=nrow):
                    if config.update_strategy is not None:
                        parameters_updatable.append(f"{key} ({ncol}, {nrow})")
                    else:
                        parameters_not_updatable.append(f"{key} ({ncol}, {nrow})")
                    count += len(config)

        parameters_updatable += [
            f"{group_name} ({cnt})"
            for group_name, cnt in genkw_groups_updadatable.items()
        ]
        parameters_not_updatable += [
            f"{group_name} ({cnt})"
            for group_name, cnt in genkw_groups_not_updadatable.items()
        ]

        return (
            sorted(parameters_updatable, key=lambda k: k.lower()),
            sorted(parameters_not_updatable, key=lambda k: k.lower()),
            count,
        )

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
