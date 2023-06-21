from __future__ import annotations

import io
import logging
from itertools import combinations as combi
from json.decoder import JSONDecodeError
from typing import List, Optional, TYPE_CHECKING

import httpx
import pandas as pd
import requests
from pandas.errors import ParserError

if TYPE_CHECKING:
    from ert.storage import StorageReader, EnsembleReader

logger = logging.getLogger(__name__)


class PlotApi:
    def __init__(self, storage: StorageReader):
        self._all_cases: List[EnsembleReader] = []
        self._timeout = 120
        self._reset_storage_facade()
        self._storage = storage

    def _reset_storage_facade(self):
        self._storage.refresh()

    def _get_all_cases(self) -> List[dict]:
        if self._all_cases is not None:
            return self._all_cases

        self._all_cases = [
            {"name": x.name, "id": x.id, "hidden": False}
            for x in self._storage.ensembles
        ]

    @staticmethod
    def _check_response(response: requests.Response):
        if response.status_code != httpx.codes.OK:
            raise httpx.RequestError(
                f" Please report this error and try restarting the application."
                f"{response.text} from url: {response.url}."
            )

    def all_data_type_keys(self) -> List:
        """Returns a list of all the keys except observation keys.

        The keys are a unique set of all keys in the ensembles

        For each key a dict is returned with info about
        the key"""

        all_keys = {}
        with StorageService.session() as client:
            for experiment in self._get_experiments():
                for ensemble in self._get_ensembles(experiment["id"]):
                    response: requests.Response = client.get(
                        f"/ensembles/{ensemble['id']}/responses", timeout=self._timeout
                    )
                    self._check_response(response)
                    for key, value in response.json().items():
                        all_keys[key] = {
                            "key": key,
                            "index_type": "VALUE",
                            "observations": value["has_observations"],
                            "dimensionality": 2,
                            "metadata": value["userdata"],
                            "log_scale": key.startswith("LOG10_"),
                        }

                    response: requests.Response = client.get(
                        f"/ensembles/{ensemble['id']}/parameters", timeout=self._timeout
                    )
                    self._check_response(response)
                    for e in response.json():
                        key = e["name"]
                        all_keys[key] = {
                            "key": key,
                            "index_type": None,
                            "observations": False,
                            "dimensionality": 1,
                            "metadata": e["userdata"],
                            "log_scale": key.startswith("LOG10_"),
                        }

        return list(all_keys.values())

    def get_all_cases_not_running(self) -> List:
        """Returns a list of all cases that are not running. For each case a dict with
        info about the case is returned"""
        # Currently, the ensemble information from the storage API does not contain any
        # hint if a case is running or not for now we return all the cases, running or
        # not
        return self._get_all_cases()

    def data_for_key(self, case_name, key) -> pd.DataFrame:
        """Returns a pandas DataFrame with the datapoints for a given key for a given
        case. The row index is the realization number, and the columns are an index
        over the indexes/dates"""

        if key.startswith("LOG10_"):
            key = key[6:]

        ensemble = self._storage.get_ensemble_by_name(case_name)
        return LibresFacade

        with StorageService.session() as client:
            response: requests.Response = client.get(
                f"/ensembles/{case['id']}/records/{key}",
                headers={"accept": "application/x-parquet"},
                timeout=self._timeout,
            )
            self._check_response(response)

            stream = io.BytesIO(response.content)
            df = pd.read_parquet(stream)

            try:
                df.columns = pd.to_datetime(df.columns)
            except (ParserError, ValueError):
                df.columns = [int(s) for s in df.columns]

            try:
                return df.astype(float)
            except ValueError:
                return df

    def observations_for_key(self, case_name, key):
        """Returns a pandas DataFrame with the datapoints for a given observation key
        for a given case. The row index is the realization number, and the column index
        is a multi-index with (obs_key, index/date, obs_index), where index/date is
        used to relate the observation to the data point it relates to, and obs_index
        is the index for the observation itself"""

        ensemble = self._storage.get_ensemble_by_name(case_name)

        with StorageService.session() as client:
            response = client.get(
                f"/ensembles/{case['id']}/records/{key}/observations",
                timeout=self._timeout,
            )
            self._check_response(response)
            try:
                obs = response.json()[0]
            except (KeyError, IndexError, JSONDecodeError):
                raise httpx.RequestError("Observation schema might have changed")
            try:
                int(obs["x_axis"][0])
                key_index = [int(v) for v in obs["x_axis"]]
            except ValueError:
                key_index = [pd.Timestamp(v) for v in obs["x_axis"]]

            data_struct = {
                "STD": obs["errors"],
                "OBS": obs["values"],
                "key_index": key_index,
            }
            return pd.DataFrame(data_struct).T

    def _add_index_range(self, data):
        """
        Adds a second column index with which corresponds to the data
        index. This is because in libres simulated data and observations
        are connected through an observation key and data index, so having
        that information available when the data is joined is helpful.
        """
        arrays = [data.columns.to_list(), list(range(len(data.columns)))]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=["key_index", "data_index"])
        data.columns = index

    def history_data(self, key, case=None) -> pd.DataFrame:
        """Returns a pandas DataFrame with the data points for the history for a
        given data key, if any.  The row index is the index/date and the column
        index is the key."""

        if ":" in key:
            head, tail = key.split(":", 2)
            history_key = f"{head}H:{tail}"
        else:
            history_key = f"{key}H"

        df = self.data_for_key(case, history_key)

        if not df.empty:
            df = df.T
            # Drop columns with equal data
            duplicate_cols = [
                cc[0] for cc in combi(df.columns, r=2) if (df[cc[0]] == df[cc[1]]).all()
            ]
            return df.drop(columns=duplicate_cols)

        return pd.DataFrame()
