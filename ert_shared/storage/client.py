import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from ert_shared.feature_toggling import feature_enabled
from ert_shared.storage.server_monitor import ServerMonitor


def convertdate(dstring):
    return datetime.strptime(dstring, "%Y-%m-%d %H:%M:%S")


class StorageClient:
    def __init__(self, base_url, auth):
        self._BASE_URI = base_url
        self._auth = auth

    def all_data_type_keys(self):
        """Returns a list of all the keys except observation keys. For each key a dict is returned with info about
            the key

        example
        result = [
            {
                "key": "param1",
                "index_type": None,
                "observations": [],
                "has_refcase": False,
                "dimensionality": 1,
                "metadata": {"data_origin": "Parameters"}
            },
            {
                "key": "response1",
                "index_type": None,
                "observations": [observation*],
                "has_refcase": False,
                "dimensionality": 2,
                "metadata": {"data_origin": "Response"}
            }
        ]
        * the observation type is a direct copy of the <observations> entry
        in a response. This contains, among others, data url for values, std
        """

        ensembles = self._ref_request("{base}/ensembles".format(base=self._BASE_URI))
        if not ensembles["ensembles"]:
            return []
        ens_schema = self._ref_request(ensembles["ensembles"][0]["ref_url"])

        def obs_for_response(ref_url):
            response = self._ref_request(ref_url)

            if "observations" not in response:
                return []

            for observation in response["observations"]:
                observation["name"] = response["name"]

            return response["observations"]

        result = [
            {
                "key": resp["name"],
                "index_type": None,
                "observations": obs_for_response(resp["ref_url"]),
                "has_refcase": False,
                "dimensionality": 2,
                "metadata": {"data_origin": "Reponse"},
                "log_scale": False,
            }
            for resp in ens_schema["responses"]
        ]

        result.extend(
            [
                {
                    "key": param["group"] + ":" + param["key"],
                    "index_type": None,
                    "observations": [],
                    "has_refcase": False,
                    "dimensionality": 1,
                    "metadata": {"data_origin": "Parameter"},
                    "log_scale": param["group"].startswith("LOG10_"),
                }
                for param in ens_schema["parameters"]
            ]
        )

        return result

    def get_all_cases_not_running(self):
        """Returns a list of all cases that are not running. For each case a dict with info about the case is
        returned

        example:
        [
            {
                'has_data': True,
                'hidden': False,
                'name': <ens_name>
            },
        ]
        """

        r = requests.get(
            "{base}/ensembles".format(base=self._BASE_URI), auth=self._auth
        )

        print(r.content)
        ensembles = r.json()["ensembles"]

        return [
            {"has_data": True, "hidden": False, "name": ens["name"]}
            for ens in ensembles
        ]

    def data_for_key(self, case, key):
        """Returns a pandas DataFrame with the datapoints for a given key for a given case. The row index is
        the realization number, and the column index is a multi-index with (key, index/date)"""

        if key.startswith("LOG10_"):
            key = key[6:]

        ensembles = self._ref_request("{base}/ensembles".format(base=self._BASE_URI))

        ens = [ens for ens in ensembles["ensembles"] if ens["name"] == case][0]
        ens_schema = self._ref_request(ens["ref_url"])

        df = pd.DataFrame()

        # Response Key
        for resp in ens_schema["responses"]:
            if resp["name"] != key:
                continue

            response = self._ref_request(resp["ref_url"])

            df = self._read_csv(response["alldata_url"], header=None)
            indexes = self._axis_request(response["axis"]["data"])
            df.columns = indexes
            break

        # Parameter key - we only check if necessary
        if df.empty:
            for param in ens_schema["parameters"]:
                if param["group"] + ":" + param["key"] != key:
                    continue

                parameter = self._ref_request(param["ref_url"])

                df = self._read_csv(parameter["alldata_url"], header=None)

        return df

    def observations_for_obs_keys(self, case, obs_keys):
        """Returns a pandas DataFrame with the datapoints for a given observation key for a given case. The row index
        is the realization number, and the column index is a multi-index with (obs_key, index/date, obs_index),
        where index/date is used to relate the observation to the data point it relates to, and obs_index is
        the index for the observation itself"""

        df = pd.DataFrame()

        if len(obs_keys) == 0:
            return df

        for obs_key in obs_keys:
            key_df = pd.DataFrame()
            std = obs_key["data"]["std"]["data"]
            values = obs_key["data"]["values"]["data"]

            key_df = key_df.append(pd.Series(values, name="OBS"))
            key_df = key_df.append(pd.Series(std, name="STD"))

            key_indexes = self._axis_request(obs_key["data"]["key_indexes"]["data"])
            data_indexes = self._axis_request(obs_key["data"]["data_indexes"]["data"])

            arrays = [[obs_key["name"]] * len(key_indexes), key_indexes, data_indexes]

            index = pd.MultiIndex.from_arrays(
                arrays, names=("obs_key", "key_index", "data_index")
            )
            key_df.columns = index
            df = pd.concat([df, key_df], axis=1)

        return df

    def refcase_data(self, key):
        """Returns a pandas DataFrame with the data points for the refcase for a given data key, if any.
        The row index is the index/date and the column index is the key."""
        return pd.DataFrame()

    def history_data(self, key, case=None):
        """Returns a pandas DataFrame with the data points for the history for a given data key, if any.
        The row index is the index/date and the column index is the key."""
        return pd.DataFrame()

    def shutdown(self):
        """A noop---the lifecycle of the server is managed by the user."""
        pass

    def _read_csv(self, data_url, **kwargs):
        resp = requests.get(data_url, auth=self._auth)
        sio = StringIO(resp.text)
        return pd.read_csv(sio, **kwargs)

    def _axis_request(self, data):
        try:
            if data and type(data[0]) is str:
                return list(map(convertdate, data))
            return data
        except ValueError as e:
            raise ValueError("Could not parse indexes as either int or dates", e)

    def _data_request(self, data_url):
        resp = requests.get(data_url, auth=self._auth)
        data = resp.content.decode(resp.encoding)
        return list(map(float, data.split(",")))

    def _ref_request(self, data_url):
        resp = requests.get(data_url, auth=self._auth)
        return resp.json()


@feature_enabled("new-storage")
def create_client():
    server = ServerMonitor.get_instance()
    return StorageClient(server.fetch_url(), server.fetch_auth())
