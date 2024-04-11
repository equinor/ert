from datetime import datetime
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Union,
)

import pandas as pd
import xarray as xr
from pydantic import BaseModel, Field

from ert.config.responses.summary_observation import SummaryObservation

from .observation_vector import ObsVector
from .response_properties import ResponseTypes


def history_key(key: str) -> str:
    keyword, *rest = key.split(":")
    return ":".join([keyword + "H"] + rest)


class _AccumulatedDataset(Protocol):
    def to_xarray(self) -> xr.Dataset: ...
    def __len__(self) -> int: ...


class _SummaryObsDataset(BaseModel):
    summary_keys: List[str] = Field(default_factory=lambda: [])
    observations: List[float] = Field(default_factory=lambda: [])
    stds: List[float] = Field(default_factory=lambda: [])
    times: List[Union[int, datetime]] = Field(default_factory=lambda: [])
    obs_names: List[str] = Field(default_factory=lambda: [])

    def __len__(self) -> int:
        return len(self.summary_keys)

    def to_xarray(self) -> xr.Dataset:
        return (
            pd.DataFrame(
                data={
                    "name": self.summary_keys,
                    "obs_name": self.obs_names,
                    "time": self.times,
                    "observations": self.observations,
                    "std": self.stds,
                },
            )
            .set_index(["name", "obs_name", "time"])
            .to_xarray()
        )


class _GenObsDataset(BaseModel):
    gen_data_keys: List[str] = Field(default_factory=lambda: [])
    observations: List[float] = Field(default_factory=lambda: [])
    stds: List[float] = Field(default_factory=lambda: [])
    indexes: List[int] = Field(default_factory=lambda: [])
    report_steps: List[int] = Field(default_factory=lambda: [])
    obs_names: List[str] = Field(default_factory=lambda: [])

    def __len__(self) -> int:
        return len(self.gen_data_keys)

    def to_xarray(self) -> xr.Dataset:
        return (
            pd.DataFrame(
                data={
                    "name": self.gen_data_keys,
                    "obs_name": self.obs_names,
                    "report_step": self.report_steps,
                    "index": self.indexes,
                    "observations": self.observations,
                    "std": self.stds,
                }
            )
            .set_index(["name", "obs_name", "report_step", "index"])
            .to_xarray()
        )


class _GenObsAccumulator:
    def __init__(self) -> None:
        self.ds: _GenObsDataset = _GenObsDataset()

    def write(
        self,
        gen_data_key: str,
        obs_name: str,
        report_step: int,
        observations: List[float],
        stds: List[float],
        indexes: List[int],
    ) -> None:
        self.ds.gen_data_keys.extend([str(gen_data_key)] * len(observations))
        self.ds.obs_names.extend([str(obs_name)] * len(observations))
        self.ds.report_steps.extend([report_step] * len(observations))

        self.ds.observations.extend(observations)
        self.ds.stds.extend(stds)
        self.ds.indexes.extend(indexes)


class _SummaryObsAccumulator:
    def __init__(self) -> None:
        self.ds: _SummaryObsDataset = _SummaryObsDataset()

    def write(
        self,
        summary_key: str,
        obs_names: List[str],
        observations: List[float],
        stds: List[float],
        times: List[Union[int, datetime]],
    ) -> None:
        self.ds.summary_keys.extend([str(summary_key)] * len(obs_names))
        self.ds.obs_names.extend(map(str, obs_names))
        self.ds.observations.extend(observations)
        self.ds.stds.extend(stds)
        self.ds.times.extend(times)


# Columns used to form a key for observations of the response_type
ObservationsIndices = {"summary": ["time"], "gen_data": ["index", "report_step"]}


class EnkfObs:
    def __init__(self, obs_vectors: Dict[str, ObsVector], obs_time: List[datetime]):
        self.obs_vectors = obs_vectors
        self.obs_time = obs_time

        vecs: List[ObsVector] = [*self.obs_vectors.values()]

        gen_obs = _GenObsAccumulator()
        sum_obs = _SummaryObsAccumulator()

        # Faster to not create a single xr.Dataset per
        # observation and then merge/concat
        # this just accumulates 1d vecs before making a dataset
        for vec in vecs:
            if vec.observation_type == ResponseTypes.GEN_DATA:
                for report_step, node in vec.observations.items():
                    # wrt typing:
                    # Goal is to deprecate the entire enkfobs, it has
                    # weird union types mimicing generic classes
                    # which is hard to prove correct to mypy
                    # so ignoring errors
                    gen_obs.write(
                        gen_data_key=vec.response_name,
                        obs_name=vec.observation_name,
                        report_step=report_step,  # type: ignore
                        observations=list(node.values),  # type: ignore
                        stds=list(node.stds),  # type: ignore
                        indexes=list(node.indices),  # type: ignore
                    )

            elif vec.observation_type == ResponseTypes.SUMMARY:
                observations = []
                stds = []
                dates = []
                obs_keys = []

                for the_date, obs in vec.observations.items():
                    assert isinstance(obs, SummaryObservation)
                    observations.append(obs.value)
                    stds.append(obs.std)
                    dates.append(the_date)
                    obs_keys.append(obs.observation_key)

                sum_obs.write(
                    summary_key=vec.response_name,
                    obs_names=obs_keys,
                    observations=observations,
                    stds=stds,
                    times=dates,
                )
            else:
                raise ValueError("Unknown observation type")

        accumulated_ds: Dict[str, _AccumulatedDataset] = {
            "gen_data": gen_obs.ds,
            "summary": sum_obs.ds,
        }

        obs_dict: Dict[str, xr.Dataset] = {}
        for key in ["gen_data", "summary"]:
            acc_ds = accumulated_ds[key]
            if len(acc_ds) > 0:
                ds = acc_ds.to_xarray()
                ds.attrs["response"] = key
                obs_dict[key] = ds

        self.datasets: Dict[str, xr.Dataset] = obs_dict

    def __len__(self) -> int:
        return len(self.obs_vectors)

    def __contains__(self, key: str) -> bool:
        return key in self.obs_vectors or key in self.datasets

    def __iter__(self) -> Iterator[ObsVector]:
        return iter(self.obs_vectors.values())

    def __getitem__(self, key: str) -> Union[ObsVector, xr.Dataset]:
        if key in self.obs_vectors:
            return self.obs_vectors[key]

        if key in self.datasets:
            return self.datasets[key]

        raise KeyError(f"No observation dataset found for key {key}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnkfObs):
            return False
        # Datasets contains the full observations, so if they are equal, everything is
        return self.datasets == other.datasets

    def __repr__(self) -> str:
        return f"EnkfObs({self.obs_vectors}, {self.obs_time})"

    def get_dataset(self, key: str) -> Optional[xr.Dataset]:
        """
        Only used as convenience for backwards compatible testing
        """
        ds_for_type = next(ds for ds in self.datasets.values() if key in ds["obs_name"])

        if not ds_for_type:
            return None

        return ds_for_type.sel(obs_name=key, drop=True)


def group_observations_by_response_type(
    obs_vectors: Dict[str, ObsVector], obs_time: List[datetime]
) -> EnkfObs:
    # Q for review:
    # deprecate EnkfObs or do we need
    # to keep the dict for LibresFacade?
    return EnkfObs(obs_vectors, obs_time)
