from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Tuple, Union

import xarray as xr

from .enkf_observation_implementation_type import EnkfObservationImplementationType
from .general_observation import GenObservation
from .summary_observation import SummaryObservation

if TYPE_CHECKING:
    from .observations import EnkfObs


@dataclass
class ObsVector:
    observation_type: EnkfObservationImplementationType
    observation_key: str
    data_key: str
    observations: Dict[int, Union[GenObservation, SummaryObservation]]

    def __iter__(self) -> Iterable[Union[SummaryObservation, GenObservation]]:
        """Iterate over active report steps; return node"""
        return iter(self.observations.values())

    def __len__(self) -> int:
        return len(self.observations)

    def to_dataset(
        self, obs: "EnkfObs", active_list: List[int]
    ) -> Tuple[str, xr.Dataset]:
        if self.observation_type == EnkfObservationImplementationType.GEN_OBS:
            datasets = []
            for time_step, node in self.observations.items():
                if active_list and time_step not in active_list:
                    continue

                assert isinstance(node, GenObservation)
                datasets.append(
                    xr.Dataset(
                        {
                            "observations": (["report_step", "index"], [node.values]),
                            "std": (["report_step", "index"], [node.stds]),
                        },
                        coords={"index": node.indices, "report_step": [time_step]},
                    )
                )
            return self.data_key, xr.combine_by_coords(datasets)  # type: ignore
        elif self.observation_type == EnkfObservationImplementationType.SUMMARY_OBS:
            observations = []
            errors = []
            active_steps = (
                active_list if active_list else list(self.observations.keys())
            )

            for time_step in active_steps:
                n = self.observations[time_step]
                assert isinstance(n, SummaryObservation)
                observations.append(n.value)
                errors.append(n.std)
            time_axis = [obs.obs_time[i] for i in active_steps]
            return "summary", xr.Dataset(
                {
                    "observations": (["name", "time"], [observations]),
                    "std": (["name", "time"], [errors]),
                },
                coords={"time": time_axis, "name": [self.observation_key]},
            )
        else:
            raise ValueError(f"Unknown observation type {self.observation_type}")
