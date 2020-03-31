import pandas as pd
import requests

def convert_response_to_float(resp):
    """
    Takes the return of a request
    """
    string = resp.content.decode(resp.encoding)
    return [float(x) for x in string.split(",")]

class StorageClient(object):
    def __init__(self):
        self._BASE_URI = "http://127.0.0.1:5000/" # Default flask server

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
                "observations": ["obs1"],
                "has_refcase": False,
                "dimensionality": 2,
                "metadata": {"data_origin": "Response"}
            }
        ]
        """

        r = requests.get("{base}/ensembles".format(base=self._BASE_URI))

        ens_pointer = r.json()["ensembles"][0]["ref_pointer"]

        r = requests.get(ens_pointer)

        ens_schema = r.json()

        result =  [
                {
                    "key": resp["name"],
                    "index_type": None,
                    "observations": [],
                    "has_refcase": False,
                    "dimensionality": 2,
                    "metadata": {"data_origin": "Reponse"},
                }
                for resp in ens_schema["responses"]
            ]

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
            {"has_data": True, "hidden": False, "name": ens["name"]} for ens in ensembles
        ]

    def data_for_key(self, case, key):
        """ Returns a pandas DataFrame with the datapoints for a given key for a given case. The row index is
            the realization number, and the column index is a multi-index with (key, index/date)"""

        r = requests.get("{base}/ensembles".format(base=self._BASE_URI))

        ens_pointer = r.json()["ensembles"][0]["ref_pointer"]

        r = requests.get(ens_pointer)

        ens_schema = r.json()

        df = pd.DataFrame()
        for real in ens_schema["realizations"]:

            r = requests.get(real["ref_pointer"])
            realization = r.json()
            response_df = pd.DataFrame()
            for resp in realization["responses"]:
                if resp["name"] != key:
                    continue

                r = requests.get(resp["data_ref"])

                # TODO: simplified for now, expected structure not in place
                # Need to add index as well

                # issue:
                # r.json() -> json.decoder.JSONDecodeError: Extra data: line 1 column 8 (char 7)
                data = pd.Series(convert_response_to_float(r), name=real["name"])

                response_df = response_df.append(data)

            df = df.append(response_df)

        arrays = [[key]*len(df.columns), df.columns]
        index = pd.MultiIndex.from_arrays(arrays, names=('key', 'index'))
        df.columns = index
        return df

    def observations_for_obs_keys(self, case, obs_keys): #is obs_keys really plural?
        """ Returns a pandas DataFrame with the datapoints for a given observation key for a given case. The row index
            is the realization number, and the column index is a multi-index with (obs_key, index/date, obs_index),
            where index/date is used to relate the observation to the data point it relates to, and obs_index is
            the index for the observation itself"""

        return pd.DataFrame()

    def refcase_data(self, key):
        """ Returns a pandas DataFrame with the data points for the refcase for a given data key, if any.
            The row index is the index/date and the column index is the key."""
        return pd.DataFrame()
