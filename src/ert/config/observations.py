from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from .observation_vector import ObsVector
from .parsing import (
    ObservationType,
)


@dataclass
class EnkfObs:
    obs_vectors: dict[str, ObsVector] = field(default_factory=dict)
    obs_time: list[datetime] = field(default_factory=list)

    def __post_init__(self) -> None:
        grouped: dict[str, list[pl.DataFrame]] = {}
        for vec in self.obs_vectors.values():
            if vec.observation_type == ObservationType.SUMMARY:
                if "summary" not in grouped:
                    grouped["summary"] = []

                grouped["summary"].append(vec.to_dataset([]))

            elif vec.observation_type == ObservationType.GENERAL:
                if "gen_data" not in grouped:
                    grouped["gen_data"] = []

                grouped["gen_data"].append(vec.to_dataset([]))

        datasets: dict[str, pl.DataFrame] = {}

        for name, dfs in grouped.items():
            non_empty_dfs = [df for df in dfs if not df.is_empty()]
            if len(non_empty_dfs) > 0:
                ds = pl.concat(non_empty_dfs).sort("observation_key")
                if "time" in ds:
                    ds = ds.sort(by="time")

                datasets[name] = ds

        self.datasets = datasets

    def __len__(self) -> int:
        return len(self.obs_vectors)

    def __contains__(self, key: str) -> bool:
        return key in self.obs_vectors

    def __iter__(self) -> Iterator[ObsVector]:
        return iter(self.obs_vectors.values())

    def __getitem__(self, key: str) -> ObsVector:
        return self.obs_vectors[key]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnkfObs):
            return False

        if self.datasets.keys() != other.datasets.keys():
            return False

        # Datasets contains the full observations, so if they are equal, everything is
        return all(self.datasets[k].equals(other.datasets[k]) for k in self.datasets)

    def __repr__(self) -> str:
        return f"EnkfObs({self.obs_vectors}, {self.obs_time})"
