class StorageClient(object):

    def __init__(self, session=None, blob_session=None):
        self._session = session
        self._blob_session = blob_session

    def _repo(self):
        if self._session is not None:
            return RdbApi(self._session)
        else:
            return RdbApi()

    def _blob(self):
        if self._session is not None:
            return BlobApi(self._session)
        else:
            return BlobApi()

    def all_data_type_keys(self):
        """ Returns a list of all the keys except observation keys. For each key a dict is returned with info about
            the key"""
        # Will likely have to change this to somehow include the ensemble name
        ens = self._repo().get_all_ensembles()[0]

        schema = self.ensemble_schema(ens.name)

        result = []

        result.extend([{
            "key": param["name"],
            "index_type": None,
            "observations": [],
            "has_refcase": False,
            "dimensionality": 1,
            "metadata": {"data_origin": "Parameters"}
        } for param in schema["parameters"]])

        result.extend([{
            "key": resp["name"],
            "index_type": None,
            "observations": resp.get("observed_by", []),
            "has_refcase": False,
            "dimensionality": 2,
            "metadata": {"data_origin": "Response"}
        } for resp in schema["responses"]])

        return result


    def get_all_cases_not_running(self):
        """ Returns a list of all cases that are not running. For each case a dict with info about the case is
            returned """

        return [{'has_data': True, 'hidden': False, 'name': ens.name} for ens in self._repo().get_all_ensembles()]


    def data_for_key(self, case, key):
        """ Returns a pandas DataFrame with the datapoints for a given key for a given case. The row index is
            the realization number, and the column index is a multi-index with (key, index/date)"""
        csv = ""
        for msg in self.stream_csv(self.get_param_data([case], [key])):
            csv += msg
        df = pd.read_csv(StringIO(csv), index_col=[0], header=[0, 1]).dropna().astype(float)
        return df


    def observations_for_obs_keys(self, case, obs_keys):
        """ Returns a pandas DataFrame with the datapoints for a given observation key for a given case. The row index
            is the realization number, and the column index is a multi-index with (obs_key, index/date, obs_index),
            where index/date is used to relate the observation to the data point it relates to, and obs_index is
            the index for the observation itself"""
        blob = self._blob()
        if len(obs_keys) > 0:
            obs = self._repo().get_observation(obs_keys[0])
            data_indexes = blob.get_blob(obs.data_indexes_ref)
            key_indexes = blob.get_blob(obs.key_indexes_ref)
            values = blob.get_blob(obs.values_ref)
            stds = blob.get_blob(obs.stds_ref)
            idx = pd.MultiIndex.from_arrays([data_indexes.data, key_indexes.data],
                                             names=['data_index', 'key_index'])

            all = pd.DataFrame({"OBS":values.data, "STD":stds.data}, index=idx)
            return all.T
        else:
            return pd.DataFrame()


    def _add_index_range(self, data):
        """
        Adds a second column index with which corresponds to the data
        index. This is because in libres simulated data and observations
        are connected through an observation key and data index, so having
        that information available when the data is joined is helpful.
        """
        pass


    def refcase_data(self, key):
        """ Returns a pandas DataFrame with the data points for the refcase for a given data key, if any.
            The row index is the index/date and the column index is the key."""
        return pd.DataFrame()