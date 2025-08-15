from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations as combi
from json.decoder import JSONDecodeError
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import quote

import httpx
import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.errors import ParserError

from ert.config import ParameterMetadata, ResponseMetadata
from ert.services import StorageService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, eq=True)
class EnsembleObject:
    name: str
    id: str
    hidden: bool
    experiment_name: str


class PlotApiKeyDefinition(NamedTuple):
    key: str
    index_type: str | None
    observations: bool
    dimensionality: int
    metadata: dict[Any, Any]
    log_scale: bool
    filter_on: dict[Any, Any] | None = None
    parameter_metadata: ParameterMetadata | None = None
    response_metadata: ResponseMetadata | None = None


def _history_key(key: str) -> str:
    """The history summary key responding to given summary key

    See :ref:`SUMMARY  <summary>` and :ref:`HISTORY_SOURCE <history_source>`
    for in keywords.rst for details.

    >>> _history_key("FOPR")
    'FOPRH'
    >>> _history_key("BPR:1,3,8")
    'BPRH:1,3,8'
    >>> _history_key("LWWIT:WNAME:LGRNAME")
    'LWWITH:WNAME:LGRNAME'
    """
    if ":" in key:
        head, tail = key.split(":", 1)
        history_key = f"{head}H:{tail}"
    else:
        history_key = f"{key}H"

    return history_key


class PlotApi:
    def __init__(self, ens_path: Path) -> None:
        self.ens_path = ens_path
        self._all_ensembles: list[EnsembleObject] | None = None
        self._timeout = 120

    @staticmethod
    def escape(s: str) -> str:
        return quote(quote(s, safe=""))

    def _get_ensemble_by_id(self, id_: str) -> EnsembleObject | None:
        for ensemble in self.get_all_ensembles():
            if ensemble.id == id_:
                return ensemble
        return None

    def get_all_ensembles(self) -> list[EnsembleObject]:
        if self._all_ensembles is not None:
            return self._all_ensembles

        self._all_ensembles = []
        with StorageService.session(project=self.ens_path) as client:
            try:
                response = client.get("/experiments", timeout=self._timeout)
                self._check_response(response)
                experiments = response.json()
                for experiment in experiments:
                    for ensemble_id in experiment["ensemble_ids"]:
                        response = client.get(
                            f"/ensembles/{ensemble_id}", timeout=self._timeout
                        )
                        self._check_response(response)
                        response_json = response.json()
                        ensemble_name: str = response_json["userdata"]["name"]
                        experiment_name: str = response_json["userdata"][
                            "experiment_name"
                        ]
                        self._all_ensembles.append(
                            EnsembleObject(
                                name=ensemble_name,
                                id=ensemble_id,
                                experiment_name=experiment_name,
                                hidden=ensemble_name.startswith("."),
                            )
                        )
            except IndexError as exc:
                logger.exception(exc)
                raise exc
            else:
                return self._all_ensembles

    @staticmethod
    def _check_response(response: httpx._models.Response) -> None:
        if response.status_code == httpx.codes.UNAUTHORIZED:
            raise httpx.RequestError(message=f"{response.text}")
        if response.status_code != httpx.codes.OK:
            raise httpx.RequestError(
                f" Please report this error and try restarting the application."
                f"{response.text} from url: {response.url}."
            )

    @cached_property
    def parameters_api_key_defs(self) -> list[PlotApiKeyDefinition]:
        all_keys: dict[str, PlotApiKeyDefinition] = {}
        all_params = {}

        with StorageService.session(project=self.ens_path) as client:
            response = client.get("/experiments", timeout=self._timeout)
            self._check_response(response)

            for experiment in response.json():
                for param_metadatas in experiment["parameters"].values():
                    for metadata in param_metadatas:
                        param_key = metadata["key"]
                        all_keys[param_key] = PlotApiKeyDefinition(
                            key=param_key,
                            index_type=None,
                            observations=False,
                            dimensionality=metadata["dimensionality"],
                            metadata=metadata["userdata"],
                            log_scale=(metadata["transformation"] or "None")
                            .lower()
                            .startswith("log"),
                            parameter_metadata=ParameterMetadata(**metadata),
                        )
                        all_params[param_key] = all_keys[param_key]

        return list(all_keys.values())

    @cached_property
    def responses_api_key_defs(self) -> list[PlotApiKeyDefinition]:
        key_defs: dict[str, PlotApiKeyDefinition] = {}

        with StorageService.session(project=self.ens_path) as client:
            response = client.get("/experiments", timeout=self._timeout)
            self._check_response(response)

            for experiment in response.json():
                for response_type, response_metadatas in experiment[
                    "responses"
                ].items():
                    for metadata in response_metadatas:
                        key = metadata["response_key"]
                        has_obs = (
                            response_type in experiment["observations"]
                            and key in experiment["observations"][response_type]
                        )
                        if metadata["filter_on"]:
                            # Only assume one filter_on, this code is to be
                            # considered a bit "temp".
                            # In general, we could create a dropdown per
                            # filter_on on the frontend side
                            for filter_key, values in metadata["filter_on"].items():
                                for v in values:
                                    subkey = f"{key}@{v}"
                                    key_defs[subkey] = PlotApiKeyDefinition(
                                        key=subkey,
                                        index_type="VALUE",
                                        observations=has_obs,
                                        dimensionality=2,
                                        metadata={
                                            "data_origin": response_type,
                                        },
                                        filter_on={filter_key: v},
                                        log_scale=False,
                                        response_metadata=ResponseMetadata(**metadata),
                                    )
                        else:
                            key_defs[key] = PlotApiKeyDefinition(
                                key=key,
                                index_type="VALUE",
                                observations=has_obs,
                                dimensionality=2,
                                metadata={"data_origin": response_type},
                                log_scale=False,
                                response_metadata=ResponseMetadata(**metadata),
                            )

        return list(key_defs.values())

    def data_for_response(
        self,
        ensemble_id: str,
        response_key: str,
        filter_on: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        with StorageService.session(project=self.ens_path) as client:
            response = client.get(
                f"/ensembles/{ensemble_id}/responses/{PlotApi.escape(response_key)}",
                headers={"accept": "application/x-parquet"},
                params={"filter_on": json.dumps(filter_on)}
                if filter_on is not None
                else None,
                timeout=self._timeout,
            )
            self._check_response(response)

            stream = io.BytesIO(response.content)
            df = pd.read_parquet(stream)

            try:
                df.columns = pd.to_datetime(df.columns, format="%Y-%m-%d %H:%M:%S")
            except (ParserError, ValueError):
                df.columns = [int(s) for s in df.columns]

            try:
                return df.astype(float)
            except ValueError:
                return df

    def data_for_parameter(self, ensemble_id: str, parameter_key: str) -> pd.DataFrame:
        with StorageService.session(project=self.ens_path) as client:
            parameter = client.get(
                f"/ensembles/{ensemble_id}/parameters/{PlotApi.escape(parameter_key)}",
                headers={"accept": "application/x-parquet"},
                timeout=self._timeout,
            )
            self._check_response(parameter)

            stream = io.BytesIO(parameter.content)
            df = pd.read_parquet(stream)

            try:
                df.columns = pd.to_datetime(df.columns, format="%Y-%m-%d %H:%M:%S")
            except (ParserError, ValueError):
                df.columns = [int(s) for s in df.columns]

            for col in df.columns:
                if is_numeric_dtype(df[col]):
                    df[col] = df[col].astype(float)
            return df

    def observations_for_key(self, ensemble_ids: list[str], key: str) -> pd.DataFrame:
        """Returns a pandas DataFrame with the datapoints for a given observation key
        for a given ensembles. The row index is the realization number, and the column
        index is a multi-index with (obs_key, index/date, obs_index), where index/date
        is used to relate the observation to the data point it relates to, and obs_index
        is the index for the observation itself"""
        all_observations = pd.DataFrame()
        for ensemble_id in ensemble_ids:
            ensemble = self._get_ensemble_by_id(ensemble_id)
            if not ensemble:
                continue

            key_def = next(
                (k for k in self.responses_api_key_defs if k.key == key), None
            )
            if not key_def:
                raise httpx.RequestError(f"Response key {key_def} not found")

            assert key_def.response_metadata is not None
            actual_response_key = key_def.response_metadata.response_key
            filter_on = key_def.filter_on
            with StorageService.session(project=self.ens_path) as client:
                response = client.get(
                    f"/ensembles/{ensemble.id}/responses/{PlotApi.escape(actual_response_key)}/observations",
                    timeout=self._timeout,
                    params={"filter_on": json.dumps(filter_on)}
                    if filter_on is not None
                    else None,
                )
                self._check_response(response)

                try:
                    observations = response.json()
                    observations_dfs = []
                    if not observations:
                        continue

                    observations[0]  # Just preserving the old logic/behavior
                    # but this should really be revised
                except (KeyError, IndexError, JSONDecodeError) as e:
                    raise httpx.RequestError(
                        f"Observation schema might have changed key={key}, "
                        f"ensemble_name={ensemble.name}, e={e}"
                    ) from e

                for obs in observations:
                    try:
                        int(obs["x_axis"][0])
                        key_index = [int(v) for v in obs["x_axis"]]
                    except ValueError:
                        key_index = [pd.Timestamp(v) for v in obs["x_axis"]]

                    observations_dfs.append(
                        pd.DataFrame(
                            {
                                "STD": obs["errors"],
                                "OBS": obs["values"],
                                "key_index": key_index,
                            }
                        )
                    )

                all_observations = pd.concat([all_observations, *observations_dfs])

        return all_observations.T

    def has_history_data(self, key: str) -> bool:
        history_key = _history_key(key)
        return any(x for x in self.responses_api_key_defs if x.key == history_key)

    def history_data(
        self,
        key: str,
        ensemble_ids: list[str] | None,
        filter_on: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Returns a pandas DataFrame with the data points for the history for a
        given data key, if any.  The row index is the index/date and the column
        index is the key."""
        if not ensemble_ids:
            return pd.DataFrame()

        for ensemble_id in ensemble_ids:
            history_key = _history_key(key)

            df = self.data_for_response(ensemble_id, history_key, filter_on)

            if not df.empty:
                df = df.T
                # Drop columns with equal data
                duplicate_cols = [
                    cc[0]
                    for cc in combi(df.columns, r=2)
                    if (df[cc[0]] == df[cc[1]]).all()
                ]
                return df.drop(columns=duplicate_cols)

    def std_dev_for_parameter(
        self, key: str, ensemble_id: str, z: int
    ) -> npt.NDArray[np.float32]:
        ensemble = self._get_ensemble_by_id(ensemble_id)
        if not ensemble:
            return np.array([])

        with StorageService.session(project=self.ens_path) as client:
            response = client.get(
                f"/ensembles/{ensemble.id}/parameters/{PlotApi.escape(key)}/std_dev",
                params={"z": z},
                timeout=self._timeout,
            )

            if response.status_code == 200:
                # Deserialize the numpy array
                return np.load(io.BytesIO(response.content))
            else:
                return np.array([])
