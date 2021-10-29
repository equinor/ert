import pandas as pd
import logging

from typing import List

from ert_data import loader as loader
from ert_data.measured import MeasuredData
from ert_shared.services import Storage

logger = logging.getLogger(__name__)


class PlotApi(object):
    def __init__(self, facade):
        self._facade = facade

    def _get_all_cases(self) -> List[str]:
        cases = []
        with Storage.session() as client:
            try:
                response = client.get("/experiments")
                experiments = response.json()
                for experiment in experiments:
                    for ensemble_id in experiment["ensemble_ids"]:
                        response = client.get(f"/ensembles/{ensemble_id}")
                        response_json = response.json()
                        case_name = response_json["userdata"]["name"]
                        cases.append(case_name)
                return cases
            except Exception as e:
                logging.exception(f"Could not retrieve case information! {str(e)}")
                return []

    def all_data_type_keys(self):
        """Returns a list of all the keys except observation keys. For each key a dict is returned with info about
        the key"""

        all_keys = self._facade.all_data_type_keys()
        log_keys = [k for k in all_keys if k.startswith("LOG10_")]

        return [
            {
                "key": key,
                "index_type": self._key_index_type(key),
                "observations": self._facade.observation_keys(key),
                "has_refcase": self._facade.has_refcase(key),
                "dimensionality": self._dimensionality_of_key(key),
                "metadata": self._metadata(key),
                "log_scale": key in log_keys,
            }
            for key in all_keys
        ]

    def _metadata(self, key):
        meta = {}
        if self._facade.is_summary_key(key):
            meta["data_origin"] = "Summary"
        elif self._facade.is_gen_data_key(key):
            meta["data_origin"] = "GEN_DATA"
        elif self._facade.is_gen_kw_key(key):
            meta["data_origin"] = "GEN_KW"
        return meta

    def get_all_cases_not_running(self) -> List:
        """Returns a list of all cases that are not running. For each case a dict with info about the case is
        returned"""
        # Currently the ensemble information from the storage API does not contain any hint if a case is running or not
        # for now we return all the cases, running or not
        cases = []

        for name in self._get_all_cases():
            case = dict(name=name, hidden=name.startswith("."))
            cases.append(case)
        return cases

    def data_for_key(self, case, key):
        """Returns a pandas DataFrame with the datapoints for a given key for a given case. The row index is
        the realization number, and the columns are an index over the indexes/dates"""

        if key.startswith("LOG10_"):
            key = key[6:]

        if self._facade.is_summary_key(key):
            data = self._facade.gather_summary_data(case, key).T
        elif self._facade.is_gen_kw_key(key):
            data = self._facade.gather_gen_kw_data(case, key)
            data.columns = pd.Index([0])
        elif self._facade.is_gen_data_key(key):
            data = self._facade.gather_gen_data_data(case, key).T
        else:
            raise ValueError("no such key {}".format(key))

        try:
            return data.astype(float)
        except ValueError:
            return data

    def observations_for_obs_keys(self, case, obs_keys):
        """Returns a pandas DataFrame with the datapoints for a given observation key for a given case. The row index
        is the realization number, and the column index is a multi-index with (obs_key, index/date, obs_index),
        where index/date is used to relate the observation to the data point it relates to, and obs_index is
        the index for the observation itself"""
        try:
            measured_data = MeasuredData(
                self._facade, obs_keys, case_name=case, load_data=False
            )
            data = measured_data.data
        except loader.ObservationError:
            data = pd.DataFrame()
        expected_keys = ["OBS", "STD"]
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Invalid type: {}, should be type: {}".format(type(data), pd.DataFrame)
            )
        elif not data.empty and not set(expected_keys).issubset(data.index):
            raise ValueError(
                "{} should be present in DataFrame index, missing: {}".format(
                    ["OBS", "STD"], set(expected_keys) - set(data.index)
                )
            )
        else:
            return data

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

    def _dimensionality_of_key(self, key):
        if self._facade.is_summary_key(key) or self._facade.is_gen_data_key(key):
            return 2
        else:
            return 1

    def _key_index_type(self, key):
        if self._facade.is_gen_data_key(key):
            return "INDEX"
        elif self._facade.is_summary_key(key):
            return "VALUE"
        else:
            return None
