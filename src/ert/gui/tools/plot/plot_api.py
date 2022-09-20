import io
import logging
from itertools import combinations as combi
from json.decoder import JSONDecodeError
from typing import List, Optional

import httpx
import pandas as pd
import requests
from pandas.errors import ParserError

from ert.services import Storage

logger = logging.getLogger(__name__)


class PlotApi:
    def __init__(self):
        self._all_cases: List[dict] = None
        self._timeout = 120
        self._reset_storage_facade()

    def _reset_storage_facade(self):
        with Storage.session() as client:
            client.post("/updates/facade", timeout=self._timeout)

    def _get_case(self, name: str) -> Optional[dict]:
        for e in self._get_all_cases():
            if e["name"] == name:
                return e
        return None

    def _get_all_cases(self) -> List[dict]:
        if self._all_cases is not None:
            return self._all_cases

        self._all_cases = []
        with Storage.session() as client:
            try:
                response = client.get("/experiments", timeout=self._timeout)
                self._check_response(response)
                experiments = response.json()
                experiment = experiments[0]
                for ensemble_id in experiment["ensemble_ids"]:
                    response = client.get(
                        f"/ensembles/{ensemble_id}", timeout=self._timeout
                    )
                    self._check_response(response)
                    response_json = response.json()
                    case_name = response_json["userdata"]["name"]
                    self._all_cases.append(
                        dict(
                            name=case_name,
                            id=ensemble_id,
                            hidden=case_name.startswith("."),
                        )
                    )
                return self._all_cases
            except IndexError as exc:
                logging.exception(exc)
                raise exc

    @staticmethod
    def _check_response(response: requests.Response):
        if response.status_code != httpx.codes.OK:
            raise httpx.RequestError(
                f" Please report this error and try restarting the application."
                f"{response.text} from url: {response.url}."
            )

    def _get_experiment(self) -> dict:
        with Storage.session() as client:
            response: requests.Response = client.get(
                "/experiments", timeout=self._timeout
            )
            self._check_response(response)
            response_json = response.json()
            return response_json[0]

    def _get_ensembles(self, experiement_id) -> List:
        with Storage.session() as client:
            response: requests.Response = client.get(
                f"/experiments/{experiement_id}/ensembles", timeout=self._timeout
            )
            self._check_response(response)
            response_json = response.json()
            return response_json

    def all_data_type_keys(self) -> List:
        """Returns a list of all the keys except observation keys.

        The keys are a unique set of all keys in the ensembles

        For each key a dict is returned with info about
        the key"""

        all_keys = {}
        experiment = self._get_experiment()

        with Storage.session() as client:
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

        case = self._get_case(case_name)

        with Storage.session() as client:
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

        case = self._get_case(case_name)

        with Storage.session() as client:
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
