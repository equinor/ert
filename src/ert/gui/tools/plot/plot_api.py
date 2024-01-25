import io
import logging
from itertools import combinations as combi
from json.decoder import JSONDecodeError
from typing import Dict, List, NamedTuple, Optional

import httpx
import pandas as pd
import requests
from pandas.errors import ParserError

from ert.services import StorageService

logger = logging.getLogger(__name__)

PlotCaseObject = NamedTuple(
    "PlotCaseObject", [("name", str), ("id", str), ("hidden", bool)]
)
PlotApiKeyDefinition = NamedTuple(
    "PlotApiKeyDefinition",
    [
        ("key", str),
        ("index_type", Optional[str]),
        ("observations", bool),
        ("dimensionality", int),
        ("metadata", dict),
        ("log_scale", bool),
    ],
)


class PlotApi:
    def __init__(self):
        self._all_cases: List[PlotCaseObject] = None
        self._timeout = 120
        self._reset_storage_facade()

    def _reset_storage_facade(self):
        with StorageService.session() as client:
            client.post("/updates/facade", timeout=self._timeout)

    def _get_case(self, name: str) -> Optional[PlotCaseObject]:
        for case in self._get_all_cases():
            if case.name == name:
                return case
        return None

    def _get_all_cases(self) -> List[PlotCaseObject]:
        if self._all_cases is not None:
            return self._all_cases

        self._all_cases = []
        with StorageService.session() as client:
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
                        case_name: str = response_json["userdata"]["name"]
                        self._all_cases.append(
                            PlotCaseObject(
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

    def all_data_type_keys(self) -> List[PlotApiKeyDefinition]:
        """Returns a list of all the keys except observation keys.

        The keys are a unique set of all keys in the ensembles

        For each key a dict is returned with info about
        the key"""

        all_keys: Dict[str, PlotApiKeyDefinition] = {}

        with StorageService.session() as client:
            response: requests.Response = client.get(
                "/experiments", timeout=self._timeout
            )
            self._check_response(response)

            for experiment in response.json():
                response: requests.Response = client.get(
                    f"/experiments/{experiment['id']}/ensembles", timeout=self._timeout
                )
                self._check_response(response)

                for ensemble in response.json():
                    response: requests.Response = client.get(
                        f"/ensembles/{ensemble['id']}/responses", timeout=self._timeout
                    )
                    self._check_response(response)
                    for key, value in response.json().items():
                        assert isinstance(key, str)
                        all_keys[key] = PlotApiKeyDefinition(
                            key=key,
                            index_type="VALUE",
                            observations=value["has_observations"],
                            dimensionality=2,
                            metadata=value["userdata"],
                            log_scale=key.startswith("LOG10_"),
                        )

                    response: requests.Response = client.get(
                        f"/ensembles/{ensemble['id']}/parameters", timeout=self._timeout
                    )
                    self._check_response(response)
                    for e in response.json():
                        key = e["name"]
                        all_keys[key] = PlotApiKeyDefinition(
                            key=key,
                            index_type=None,
                            observations=False,
                            dimensionality=1,
                            metadata=e["userdata"],
                            log_scale=key.startswith("LOG10_"),
                        )

        return list(all_keys.values())

    def get_all_cases_not_running(self) -> List[PlotCaseObject]:
        """Returns a list of all cases that are not running. For each case a dict with
        info about the case is returned"""
        # Currently, the ensemble information from the storage API does not contain any
        # hint if a case is running or not for now we return all the cases, running or
        # not
        return self._get_all_cases()

    def data_for_key(self, case_name: str, key: str) -> pd.DataFrame:
        """Returns a pandas DataFrame with the datapoints for a given key for a given
        case. The row index is the realization number, and the columns are an index
        over the indexes/dates"""

        if key.startswith("LOG10_"):
            key = key[6:]

        case = self._get_case(case_name)

        with StorageService.session() as client:
            response: requests.Response = client.get(
                f"/ensembles/{case.id}/records/{key}",
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

        with StorageService.session() as client:
            response = client.get(
                f"/ensembles/{case.id}/records/{key}/observations",
                timeout=self._timeout,
            )
            self._check_response(response)
            try:
                obs = response.json()[0]
            except (KeyError, IndexError, JSONDecodeError) as e:
                raise httpx.RequestError("Observation schema might have changed") from e
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
