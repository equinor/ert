import pandas as pd
import logging
import requests
import io
import httpx
from typing import List
from ert_data import loader as loader
from ert_shared.services import Storage
from pandas.errors import ParserError

logger = logging.getLogger(__name__)


class PlotApi(object):
    def __init__(self, facade):
        self._facade = facade
        self._all_cases: List[dict] = None

    def _get_case(self, name: str) -> dict:
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
                response = client.get("/experiments")
                experiments = response.json()
                experiment = experiments[0]
                for ensemble_id in experiment["ensemble_ids"]:
                    response = client.get(f"/ensembles/{ensemble_id}")
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
            except (httpx.RequestError, IndexError) as exc:
                logging.exception(exc)
                raise exc

    def _get_experiment(self) -> dict:
        with Storage.session() as client:
            try:
                response: requests.Response = client.get("/experiments")
                response_json = response.json()
                return response_json[0]
            except (httpx.RequestError, IndexError) as exc:
                logger.exception(exc)
                raise exc

    def _get_ensembles(self, experiement_id) -> List:
        with Storage.session() as client:
            try:
                response: requests.Response = client.get(
                    f"/experiments/{experiement_id}/ensembles"
                )
                response_json = response.json()
                return response_json
            except httpx.RequestError as exc:
                logger.exception(exc)
                raise exc

    def all_data_type_keys(self) -> List:
        """Returns a list of all the keys except observation keys.

        The keys are a unique set of all keys in the enseblems

        For each key a dict is returned with info about
        the key"""

        all_keys = dict()
        experiment = self._get_experiment()

        with Storage.session() as client:

            for ensemble in self._get_ensembles(experiment["id"]):
                try:
                    response: requests.Response = client.get(
                        f"/ensembles/{ensemble['id']}/responses", timeout=None
                    )
                    for key, value in response.json().items():
                        user_data = value["userdata"]
                        has_observations = value["has_observations"]
                        all_keys[key] = {
                            "key": key,
                            "index_type": "VALUE",
                            "observations": has_observations,
                            "has_refcase": self._facade.has_refcase(key),
                            "dimensionality": 2,
                            "metadata": user_data,
                            "log_scale": key.startswith("LOG10_"),
                        }
                except httpx.RequestError as exc:
                    logger.exception(exc)
                    raise exc

                try:
                    response: requests.Response = client.get(
                        f"/ensembles/{ensemble['id']}/parameters"
                    )
                    for e in response.json():
                        key = e["name"]
                        user_data = e["userdata"]
                        all_keys[key] = {
                            "key": key,
                            "index_type": None,
                            "observations": False,
                            "has_refcase": False,
                            "dimensionality": 1,
                            "metadata": user_data,
                            "log_scale": key.startswith("LOG10_"),
                        }
                except httpx.RequestError as exc:
                    logger.exception(exc)
                    raise exc

        return list(all_keys.values())

    def get_all_cases_not_running(self) -> List:
        """Returns a list of all cases that are not running. For each case a dict with info about the case is
        returned"""
        # Currently, the ensemble information from the storage API does not contain any hint if a case is running or not
        # for now we return all the cases, running or not
        return self._get_all_cases()

    def data_for_key(self, case_name, key) -> pd.DataFrame:
        """Returns a pandas DataFrame with the datapoints for a given key for a given case. The row index is
        the realization number, and the columns are an index over the indexes/dates"""

        if key.startswith("LOG10_"):
            key = key[6:]

        case = self._get_case(case_name)

        with Storage.session() as client:
            try:
                response: requests.Response = client.get(
                    f"/ensembles/{case['id']}/records/{key}",
                    headers={"accept": "application/x-parquet"},
                )

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

            except httpx.RequestError as exc:
                logger.exception(exc)
                raise exc

    def observations_for_key(self, case_name, key):
        """Returns a pandas DataFrame with the datapoints for a given observation key for a given case. The row index
        is the realization number, and the column index is a multi-index with (obs_key, index/date, obs_index),
        where index/date is used to relate the observation to the data point it relates to, and obs_index is
        the index for the observation itself"""

        case = self._get_case(case_name)

        with Storage.session() as client:
            try:
                resp = client.get(
                    f"/ensembles/{case['id']}/records/{key}/observations"
                ).json()
                obs = resp[0]
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

            except httpx.RequestError as exc:
                logger.exception(exc)
                raise exc

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

    def refcase_data(self, key):
        """Returns a pandas DataFrame with the data points for the refcase for a given data key, if any.
        The row index is the index/date and the column index is the key."""
        return self._facade.refcase_data(key)

    def history_data(self, key, case=None):
        """Returns a pandas DataFrame with the data points for the history for a given data key, if any.
        The row index is the index/date and the column index is the key."""
        return self._facade.history_data(key, case)
