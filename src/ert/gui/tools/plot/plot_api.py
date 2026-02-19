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
from resfo_utilities import history_key

from ert.config import ParameterConfig
from ert.config.ensemble_config import ResponseConfig
from ert.config.known_response_types import KnownResponseTypes
from ert.services import create_ertserver_client
from ert.storage.local_experiment import _parameters_adapter as parameter_config_adapter
from ert.storage.local_experiment import _responses_adapter as response_config_adapter
from ert.storage.realization_storage_state import RealizationStorageState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, eq=True)
class EnsembleObject:
    name: str
    id: str
    hidden: bool
    experiment_name: str
    started_at: str


class PlotApiKeyDefinition(NamedTuple):
    key: str
    index_type: str | None
    observations: bool
    dimensionality: int
    metadata: dict[Any, Any]
    filter_on: dict[Any, Any] | None = None
    parameter: ParameterConfig | None = None
    response: ResponseConfig | None = None


class PlotApi:
    def __init__(self, ens_path: Path) -> None:
        self.ens_path: Path = ens_path
        self._all_ensembles: list[EnsembleObject] | None = None
        self._timeout = 120

    @property
    def api_version(self) -> str:
        with create_ertserver_client(self.ens_path) as client:
            try:
                http_response = client.get("/version", timeout=self._timeout)
                self._check_http_response(http_response)
                api_version = str(http_response.json())
            except Exception as exc:
                logger.exception(exc)
                raise exc
            else:
                return api_version

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
        with create_ertserver_client(self.ens_path) as client:
            try:
                http_response = client.get("/experiments", timeout=self._timeout)
                self._check_http_response(http_response)
                experiments = http_response.json()
                for experiment in experiments:
                    for ensemble_id in experiment["ensemble_ids"]:
                        http_response = client.get(
                            f"/ensembles/{ensemble_id}", timeout=self._timeout
                        )
                        self._check_http_response(http_response)
                        response_json: dict[str, Any] = http_response.json()
                        ensemble_name: str = response_json["userdata"]["name"]
                        experiment_name: str = response_json["userdata"][
                            "experiment_name"
                        ]
                        ensemble_started_at = response_json["userdata"]["started_at"]
                        ensemble_undefined = False
                        if realization_storage_states := response_json.get(
                            "realization_storage_states"
                        ):
                            ensemble_undefined = (
                                RealizationStorageState.PARAMETERS_LOADED
                                not in set(realization_storage_states)
                            )
                        self._all_ensembles.append(
                            EnsembleObject(
                                name=ensemble_name,
                                id=ensemble_id,
                                experiment_name=experiment_name,
                                hidden=ensemble_name.startswith(".")
                                or ensemble_undefined,
                                started_at=ensemble_started_at,
                            )
                        )
            except IndexError as exc:
                logger.exception(exc)
                raise exc
            else:
                return self._all_ensembles

    @staticmethod
    def _check_http_response(http_response: httpx._models.Response) -> None:
        if http_response.status_code == httpx.codes.UNAUTHORIZED:
            raise httpx.RequestError(message=f"{http_response.text}")
        if http_response.status_code != httpx.codes.OK:
            raise httpx.RequestError(
                f" Please report this error and try restarting the application."
                f"{http_response.text} from url: {http_response.url}."
            )

    @cached_property
    def parameters_api_key_defs(self) -> list[PlotApiKeyDefinition]:
        all_keys: dict[str, PlotApiKeyDefinition] = {}
        all_params = {}

        with create_ertserver_client(self.ens_path) as client:
            http_response = client.get("/experiments", timeout=self._timeout)
            self._check_http_response(http_response)

            for experiment in http_response.json():
                for metadata in experiment["parameters"].values():
                    param_cfg = parameter_config_adapter.validate_python(metadata)
                    if group := metadata.get("group"):
                        param_key = f"{group}:{metadata['name']}"
                    else:
                        param_key = metadata["name"]
                    all_keys[param_key] = PlotApiKeyDefinition(
                        key=param_key,
                        index_type=None,
                        observations=False,
                        dimensionality=metadata["dimensionality"],
                        metadata={"data_origin": metadata["type"]},
                        parameter=param_cfg,
                    )
                    all_params[param_key] = all_keys[param_key]

        return list(all_keys.values())

    @cached_property
    def responses_api_key_defs(self) -> list[PlotApiKeyDefinition]:
        key_defs: dict[str, PlotApiKeyDefinition] = {}

        with create_ertserver_client(self.ens_path) as client:
            http_response = client.get("/experiments", timeout=self._timeout)
            self._check_http_response(http_response)

            def update_keydef(plot_key_def: PlotApiKeyDefinition) -> None:
                # Only replace existing key definition if the new has observations
                if plot_key_def.key not in key_defs or plot_key_def.observations:
                    key_defs[plot_key_def.key] = plot_key_def

            for experiment in http_response.json():
                for response_type, metadata in experiment["responses"].items():
                    response_config: KnownResponseTypes = (
                        response_config_adapter.validate_python(metadata)
                    )
                    keys = response_config.keys
                    for key in keys:
                        has_obs = (
                            response_type in experiment["observations"]
                            and key in experiment["observations"][response_type]
                        )
                        if response_config.filter_on is not None:
                            # Only assume one filter_on, this code is to be
                            # considered a bit "temp".
                            # In general, we could create a dropdown per
                            # filter_on on the frontend side

                            filter_for_key = response_config.filter_on.get(key, {})
                            for filter_key, values in filter_for_key.items():
                                for v in values:
                                    filter_on = {filter_key: v}
                                    subkey = f"{key}@{v}"
                                    update_keydef(
                                        PlotApiKeyDefinition(
                                            key=subkey,
                                            index_type="VALUE",
                                            observations=has_obs,
                                            dimensionality=2,
                                            metadata={
                                                "data_origin": response_type,
                                            },
                                            filter_on=filter_on,
                                            response=response_config,
                                        )
                                    )
                        else:
                            update_keydef(
                                PlotApiKeyDefinition(
                                    key=key,
                                    index_type="VALUE",
                                    observations=has_obs,
                                    dimensionality=2,
                                    metadata={"data_origin": response_type},
                                    response=response_config,
                                )
                            )

                if "everest_objectives" in experiment["responses"]:
                    update_keydef(
                        PlotApiKeyDefinition(
                            key="total objective value",
                            index_type="VALUE",
                            observations=False,
                            dimensionality=2,
                            metadata={"data_origin": "everest_batch_objectives"},
                        )
                    )

        return list(key_defs.values())

    def data_for_response(
        self,
        ensemble_id: str,
        response_key: str,
        filter_on: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        key_def = next(
            (k for k in self.responses_api_key_defs if k.key == response_key), None
        )
        is_everest = key_def is not None and key_def.metadata.get("data_origin") in {
            "everest_objectives",
            "everest_constraints",
        }

        if "@" in response_key:
            response_key = response_key.split("@", maxsplit=1)[0]
        with create_ertserver_client(self.ens_path) as client:
            http_response = client.get(
                f"/ensembles/{ensemble_id}/responses/{PlotApi.escape(response_key)}",
                headers={"accept": "application/x-parquet"},
                params={"filter_on": json.dumps(filter_on)}
                if filter_on is not None
                else None,
                timeout=self._timeout,
            )
            self._check_http_response(http_response)

            stream = io.BytesIO(http_response.content)
            df = pd.read_parquet(stream)

            if is_everest:
                assert {"batch_id", "realization"}.issubset(df.columns)

                float_columns = [
                    col for col in df.columns if col not in {"batch_id", "realization"}
                ]

                return df.astype(
                    dict.fromkeys(float_columns, float)
                    | {
                        "batch_id": int,
                        "realization": int,
                    }
                )

            if (
                key_def is not None
                and key_def.metadata.get("data_origin") == "everest_batch_objectives"
            ):
                assert {"batch_id", "is_improvement"}.issubset(df.columns)

                float_columns = [
                    col
                    for col in df.columns
                    if col not in {"batch_id", "is_improvement"}
                ]

                return df.astype(
                    dict.fromkeys(float_columns, float)
                    | {
                        "batch_id": int,
                        "is_improvement": bool,
                    }
                )

            try:
                df.columns = pd.to_datetime(df.columns, format="%Y-%m-%d %H:%M:%S")
            except (ParserError, ValueError):
                try:
                    df.columns = [int(s) for s in df.columns]
                except ValueError:
                    df.columns = [float(s) for s in df.columns]

            try:
                return df.astype(float)
            except ValueError:
                return df

    def data_for_gradient(self, ensemble_id: str, key: str) -> pd.DataFrame:
        if "@" in key:
            key = key.split("@", maxsplit=1)[0]
        with create_ertserver_client(self.ens_path) as client:
            http_response = client.get(
                f"/ensembles/{ensemble_id}/gradients/{PlotApi.escape(key)}",
                headers={"accept": "application/x-parquet"},
                timeout=self._timeout,
            )
            self._check_http_response(http_response)

            stream = io.BytesIO(http_response.content)
            df = pd.read_parquet(stream)

            return df.astype(
                {
                    "batch_id": int,
                    "control_name": str,
                    key: float,
                }
            )

    def data_for_parameter(self, ensemble_id: str, parameter_key: str) -> pd.DataFrame:
        with create_ertserver_client(self.ens_path) as client:
            http_response = client.get(
                f"/ensembles/{ensemble_id}/parameters/{PlotApi.escape(parameter_key)}",
                headers={"accept": "application/x-parquet"},
                timeout=self._timeout,
            )
            self._check_http_response(http_response)

            stream = io.BytesIO(http_response.content)
            df = pd.read_parquet(stream)

            if {"batch_id", "realization"}.issubset(df.columns):
                return df

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
            assert key_def.response is not None
            actual_response_key = key
            if "@" in actual_response_key:
                actual_response_key = key.split("@", maxsplit=1)[0]
            filter_on = key_def.filter_on
            with create_ertserver_client(self.ens_path) as client:
                http_response = client.get(
                    f"/ensembles/{ensemble.id}/responses/{PlotApi.escape(actual_response_key)}/observations",
                    timeout=self._timeout,
                    params={"filter_on": json.dumps(filter_on)}
                    if filter_on is not None
                    else None,
                )
                self._check_http_response(http_response)

                try:
                    observations = http_response.json()
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

                key_index: list[int | float | pd.Timestamp]
                for obs in observations:
                    try:
                        int(obs["x_axis"][0])
                        key_index = [int(v) for v in obs["x_axis"]]
                    except ValueError:
                        try:
                            float(obs["x_axis"][0])
                            key_index = [float(v) for v in obs["x_axis"]]
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
        history = history_key(key)
        return any(x for x in self.responses_api_key_defs if x.key == history)

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
            history = history_key(key)

            df = self.data_for_response(ensemble_id, history, filter_on)

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

        with create_ertserver_client(self.ens_path) as client:
            http_response = client.get(
                f"/ensembles/{ensemble.id}/parameters/{PlotApi.escape(key)}/std_dev",
                params={"z": z},
                timeout=self._timeout,
            )

            if http_response.status_code == 200:
                # Deserialize the numpy array
                return np.load(io.BytesIO(http_response.content))
            else:
                return np.array([])
