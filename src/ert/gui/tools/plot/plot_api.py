import io
import logging
import os
from itertools import combinations as combi
from json.decoder import JSONDecodeError
from typing import Dict, List, NamedTuple, Optional

import httpx
import pandas as pd
from pandas.errors import ParserError

from ert.gui.ertnotifier import ErtNotifier
from ert.services import StorageService
from ert.storage import LocalStorage

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

use_new_storage = os.getenv("PLOTTER_SKIP_DARK_STORAGE") == "1"

ert_kind_to_userdata = {
    "GenDataConfig": "GEN_DATA",
    "SummaryConfig": "Summary",
    "KenKWConfig": "GEN_KW",
}


class PlotApi:
    def __init__(self, notifier: ErtNotifier):
        self._all_cases: Optional[List[PlotCaseObject]] = None
        self._timeout = 120
        self.notifier = notifier

    @property
    def storage(self):
        return self.notifier.storage

    def _get_case(self, name: str) -> Optional[PlotCaseObject]:
        for case in self._get_all_cases():
            if case.name == name:
                return case
        return None

    def _get_all_cases(self) -> List[PlotCaseObject]:
        if self._all_cases is not None:
            return self._all_cases

        if use_new_storage:
            all_cases_from_storage = []
            for exp in self.storage.experiments:
                for ens in exp.ensembles:
                    all_cases_from_storage.append(
                        PlotCaseObject(
                            name=ens.name,
                            id=str(ens.id),
                            hidden=ens.name.startswith("."),
                        )
                    )

            self._all_cases = all_cases_from_storage
            return all_cases_from_storage

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
    def _check_response(response: httpx._models.Response) -> None:
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

        if use_new_storage:
            for exp in self.storage.experiments:
                obs_keys = []
                for k, ds in exp.observations.items():
                    if "report_step" in ds.coords:
                        rslist = ds.report_step.values.tolist()
                        obs_keys.append(f"{ds.response}@{','.join(map(str,rslist))}")
                    elif ds.response == "summary":
                        kw = ds.name[0].values.tolist()
                        obs_keys.append(kw)
                    else:
                        obs_keys.append(k)

                if "summary" in exp.response_info:
                    for summary_key in exp.response_info["summary"]["keys"]:
                        all_keys[summary_key] = PlotApiKeyDefinition(
                            key=summary_key,
                            index_type="VALUE",
                            observations=summary_key in obs_keys,
                            dimensionality=2,
                            metadata={"data_origin": "Summary"},
                            log_scale=summary_key.startswith("LOG10_"),
                        )

                for key, info in [
                    (k, v) for k, v in exp.response_info.items() if k != "summary"
                ]:
                    kind = info["_ert_kind"]
                    userdata_kind = ert_kind_to_userdata[kind]
                    report_step_suffix = (
                        ",".join(map(str, info["report_steps"]))
                        if "report_steps" in info
                        else ""
                    )
                    use_key = f"{key}@{report_step_suffix}"
                    all_keys[use_key] = PlotApiKeyDefinition(
                        key=use_key,
                        index_type="VALUE",
                        observations=use_key in obs_keys,
                        dimensionality=2,
                        metadata={"data_origin": userdata_kind},
                        log_scale=use_key.startswith("LOG10_"),
                    )

                for group_name, group in exp.parameter_info.items():
                    for tf_def in group["transfer_function_definitions"]:
                        tf_key = tf_def.split(" ")[0]
                        key = f"{group_name}:{tf_key}"
                        all_keys[key] = PlotApiKeyDefinition(
                            key=key,
                            index_type=None,
                            observations=False,
                            dimensionality=1,
                            metadata={"data_origin": "GEN_KW"},
                            log_scale=key.startswith("LOG10_"),
                        )

                return list(all_keys.values())

        with StorageService.session() as client:
            response = client.get("/experiments", timeout=self._timeout)
            self._check_response(response)

            for experiment in response.json():
                response = client.get(
                    f"/experiments/{experiment['id']}/ensembles", timeout=self._timeout
                )
                self._check_response(response)

                for ensemble in response.json():
                    response = client.get(
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

                    response = client.get(
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
        if not case:
            return pd.DataFrame()

        with StorageService.session() as client:
            response = client.get(
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

    def observations_for_key(self, case_name, key) -> pd.DataFrame:
        """Returns a pandas DataFrame with the datapoints for a given observation key
        for a given case. The row index is the realization number, and the column index
        is a multi-index with (obs_key, index/date, obs_index), where index/date is
        used to relate the observation to the data point it relates to, and obs_index
        is the index for the observation itself"""

        case = self._get_case(case_name)
        if not case:
            return pd.DataFrame()

        with StorageService.session() as client:
            response = client.get(
                f"/ensembles/{case.id}/records/{key}/observations",
                timeout=self._timeout,
            )
            self._check_response(response)
            if not response.json():
                return pd.DataFrame()
            try:
                obs = response.json()[0]
            except (KeyError, IndexError, JSONDecodeError) as e:
                raise httpx.RequestError(
                    f"Observation schema might have changed key={key},  case_name={case_name}, e={e}"
                ) from e
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
