import pandas as pd
import requests
from datetime import datetime


def convertdate(dstring):
    return datetime.strptime(dstring, "%Y-%m-%d %H:%M:%S")


def axis_request(data_url):
    resp = requests.get(data_url)
    indexes = resp.content.decode(resp.encoding).split(",")
    try:
        if indexes and ":" in indexes[0]:
            return list(map(convertdate, indexes))
        else:
            return list(map(int, indexes))
    except ValueError as e:
        raise ValueError("Could not parse indexes as either int or dates", e)


def data_request(data_url):
    resp = requests.get(data_url)
    data = resp.content.decode(resp.encoding)
    return list(map(float, data.split(",")))


def ref_request(api_url):
    resp = requests.get(api_url)
    return resp.json()


class StorageClient(object):
    def __init__(self, base_url="http://127.0.0.1:5000/"):
        self._BASE_URI = base_url

    def all_data_type_keys(self):
        """ Returns a list of all the keys except observation keys. For each key a dict is returned with info about
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

        ensembles = ref_request("{base}/ensembles".format(base=self._BASE_URI))
        ens_schema = ref_request(ensembles["ensembles"][0]["ref_url"])

        def obs_for_response(ref_url):
            response = ref_request(ref_url)

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
            }
            for resp in ens_schema["responses"]
        ]

        result.extend(
            [
                {
                    "key": param["name"],
                    "index_type": None,
                    "observations": [],
                    "has_refcase": False,
                    "dimensionality": 1,
                    "metadata": {"data_origin": "Parameter"},
                }
                for param in ens_schema["parameters"]
            ]
        )

        return result

    def get_all_cases_not_running(self):
        """ Returns a list of all cases that are not running. For each case a dict with info about the case is
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

        r = requests.get("{base}/ensembles".format(base=self._BASE_URI))

        ensembles = r.json()["ensembles"]

        return [
            {"has_data": True, "hidden": False, "name": ens["name"]}
            for ens in ensembles
        ]

    def data_for_key(self, case, key):
        """ Returns a pandas DataFrame with the datapoints for a given key for a given case. The row index is
            the realization number, and the column index is a multi-index with (key, index/date)"""

        ensembles = ref_request("{base}/ensembles".format(base=self._BASE_URI))

        ens = [ens for ens in ensembles["ensembles"] if ens["name"] == case][0]
        ens_schema = ref_request(ens["ref_url"])

        df = pd.DataFrame()

        # Response Key
        for resp in ens_schema["responses"]:
            if resp["name"] != key:
                continue

            response = ref_request(resp["ref_url"])

            for real in response["realizations"]:

                data = pd.Series(data_request(real["data_url"]), name=real["name"])
                df = df.append(data)

            indexes = axis_request(response["axis"]["data_url"])
            arrays = [[key] * len(indexes), indexes]
            break

        # Parameter key - we only check if necessary
        if df.empty:
            for real in ens_schema["realizations"]:
                realization = ref_request(real["ref_url"])

                for param in realization["parameters"]:
                    if param["name"] != key:
                        continue

                    data = pd.Series(data_request(param["data_url"]), name=real["name"])
                    df = df.append(data)

            arrays = [[key] * len(df.columns), df.columns]

        index = pd.MultiIndex.from_arrays(arrays, names=("key", "index"))
        df.columns = index
        return df

    def observations_for_obs_keys(self, case, obs_keys):
        """ Returns a pandas DataFrame with the datapoints for a given observation key for a given case. The row index
            is the realization number, and the column index is a multi-index with (obs_key, index/date, obs_index),
            where index/date is used to relate the observation to the data point it relates to, and obs_index is
            the index for the observation itself"""

        df = pd.DataFrame()

        if len(obs_keys) == 0:
            return df

        for obs_key in obs_keys:
            key_df = pd.DataFrame()
            std = data_request(obs_key["data"]["std"]["data_url"])
            values = data_request(obs_key["data"]["values"]["data_url"])

            key_df = key_df.append(pd.Series(values, name="OBS"))
            key_df = key_df.append(pd.Series(std, name="STD"))

            key_indexes = axis_request(obs_key["data"]["key_indexes"]["data_url"])
            data_indexes = axis_request(obs_key["data"]["data_indexes"]["data_url"])

            arrays = [[obs_key["name"]] * len(key_indexes), key_indexes, data_indexes]

            index = pd.MultiIndex.from_arrays(
                arrays, names=("obs_key", "key_index", "data_index")
            )
            key_df.columns = index
            df = pd.concat([df, key_df], axis=1)

        return df

    def refcase_data(self, key):
        """ Returns a pandas DataFrame with the data points for the refcase for a given data key, if any.
            The row index is the index/date and the column index is the key."""
        return pd.DataFrame()
