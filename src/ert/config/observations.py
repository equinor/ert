from datetime import datetime
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

import pandas as pd
import xarray as xr

from ert.config.summary_observation import SummaryObservation

from .general_observation import GenObservation
from .observation_vector import ObsVector
from .response_properties import ResponseTypes


def history_key(key: str) -> str:
    keyword, *rest = key.split(":")
    return ":".join([keyword + "H"] + rest)


class _ObsDataset:
    def __init__(self, primary_key: List[str]):
        self._primary_key_values: Dict[str, Any] = {k: [] for k in primary_key}

        self._response_names: List[str] = []
        self._obs_names: List[str] = []
        self._observations: List[float] = []
        self._stds: List[float] = []

    def write(
        self,
        obs_name: List[str],
        response_name: List[str],
        observations: List[float],
        stds: List[float],
        primary_keys: Dict[str, List[Any]],
    ) -> None:
        self._response_names.extend(response_name)
        self._obs_names.extend(obs_name)

        self._observations.extend(observations)
        self._stds.extend(stds)

        for k, vals in primary_keys.items():
            self._primary_key_values[k].extend(vals)

    def to_xarray(self) -> xr.Dataset:
        return (
            pd.DataFrame(
                data={
                    "name": self._response_names,
                    "obs_name": self._obs_names,
                    **self._primary_key_values,
                    "observations": self._observations,
                    "std": self._stds,
                }
            )
            .set_index(["name", "obs_name", *self._primary_key_values.keys()])
            .to_xarray()
        )

    def __len__(self) -> int:
        return len(self._observations)


# Columns used to form a key for observations of the response_type
ObservationsIndices = {
    ResponseTypes.summary: ["time"],
    ResponseTypes.gen_data: ["index", "report_step"],
}


class EnkfObs:
    def __init__(self, obs_vectors: Dict[str, ObsVector], obs_time: List[datetime]):
        self.obs_vectors = obs_vectors
        self.obs_time = obs_time

        vecs: List[ObsVector] = [*self.obs_vectors.values()]

        gen_obs_ds = _ObsDataset(primary_key=["index", "report_step"])
        sum_obs_ds = _ObsDataset(primary_key=["time"])

        # Faster to not create a single xr.Dataset per
        # observation and then merge/concat
        # this just accumulates 1d vecs before making a dataset
        for vec in vecs:
            if vec.observation_type == ResponseTypes.gen_data:
                for report_step, node in vec.observations.items():
                    assert isinstance(node, GenObservation)
                    gen_obs_ds.write(
                        response_name=[vec.response_name] * len(node.values),
                        obs_name=[vec.observation_name] * len(node.values),
                        observations=list(node.values),
                        stds=list(node.stds),
                        primary_keys={
                            "report_step": [report_step] * len(node.values),
                            "index": list(node.indices),
                        },
                    )

            elif vec.observation_type == ResponseTypes.summary:
                observations = []
                stds = []
                dates = []
                obs_keys = []

                for timestamp, obs in vec.observations.items():
                    assert isinstance(obs, SummaryObservation)
                    observations.append(obs.value)
                    stds.append(obs.std)
                    dates.append(timestamp)
                    obs_keys.append(obs.observation_key)

                sum_obs_ds.write(
                    response_name=[vec.response_name] * len(observations),
                    obs_name=obs_keys,
                    observations=observations,
                    stds=stds,
                    primary_keys={"time": dates},
                )
            else:
                raise ValueError("Unknown observation type")

        accumulated_ds: Dict[str, _ObsDataset] = {
            ResponseTypes.gen_data: gen_obs_ds,
            ResponseTypes.summary: sum_obs_ds,
        }

        obs_dict: Dict[str, xr.Dataset] = {}
        for key in [ResponseTypes.summary, ResponseTypes.gen_data]:
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
    return EnkfObs(obs_vectors, obs_time)
