from io import StringIO

import pandas as pd
from ert_shared.storage.rdb_api import RdbApi


def get_response_data(name, ensemble_name, rdb_api=None):
    if rdb_api is None:
        rdb_api = RdbApi()

    with rdb_api:
        for response in rdb_api.get_response_data(name, ensemble_name):
            yield response


def get_all_ensembles(rdb_api=None):
    if rdb_api is None:
        rdb_api = RdbApi()

    with rdb_api:
        return [ensemble.name for ensemble in rdb_api.get_all_ensembles()]


class PlotStorageApi(object):

    def __init__(self, session=None):
        self._session = session

    def _repo(self):
        if self._session is not None:
            return RdbApi(self._session)
        else:
            return RdbApi()

    def all_data_type_keys(self):
        """ Returns a list of all the keys except observation keys. For each key a dict is returned with info about
            the key"""
        # Will likely have to change this to somehow include the ensemble name
        ens = self._repo().get_all_ensembles()[0]

        result = []

        result.extend([{
            "key": param.name,
            "index_type": None,
            "observations": [],
            "has_refcase": False,
            "dimensionality": 1,
            "metadata": {"data_origin": "Parameters"}
        } for param in ens.parameter_definitions])

        def _obs_names(obs):
            if obs is None:
                return []
            return [obs.name]

        result.extend([{
            "key": resp.name,
            "index_type": None,
            "observations": _obs_names(resp.observation),
            "has_refcase": False,
            "dimensionality": 2,
            "metadata": {"data_origin": "Response"}
        } for resp in ens.response_definitions])

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
        if len(obs_keys) > 0:
            obs = self._repo().get_observation(obs_keys[0])
            idx = pd.MultiIndex.from_arrays([obs.data_indexes, obs.key_indexes],
                                             names=['data_index', 'key_index'])

            all = pd.DataFrame({"OBS":obs.values, "STD":obs.stds}, index=idx)
            return all.T
        else:
            return pd.DataFrame()

    def stream_csv(self, msgs):
        done = False
        first_header = True
        first_index = True
        while not done:
            msg = next(msgs)
            if msg.type == "key_def" and msg.status == "size":
                if not first_header:
                    yield ","
                else:
                    first_header = False
                yield ",".join([msg.name for x in range(0, msg.value)])
            if msg.type == "index_def" and msg.status == "indexes":
                if not first_index:
                    yield ","
                else:
                    yield "\nindexes,"
                    first_index = False
                yield ",".join([str(x) for x in msg.value])
            if msg.type == "ensemble" and msg.status == "start":
                yield "keys,"
                pass
            elif msg.type == "realisation" and msg.status == "start":
                real_done = False
                yield "\n{},".format(msg.name)

                first_key = True
                while not real_done:
                    msg = next(msgs)
                    if msg.type == "realisation" and msg.status == "done":
                        real_done = True
                    elif msg.type in ["parameter", "response"] and msg.status == "start":
                        if not first_key:
                            yield ","
                        else:
                            first_key = False
                        param_done = False
                        while not param_done:
                            msg = next(msgs)
                            if msg.type in ["parameter", "response"] and msg.status == "chunk":

                                yield ",".join([str(x) for x in msg.value])
                            elif msg.type in ["parameter", "response"] and msg.status == "done":
                                # yield ","
                                param_done = True
            elif msg.type == "ensemble" and msg.status == "stop":
                pass
            elif msg.type == "EOL":
                done = True

    #
    # def get_obs_data(self, ensembles, obs_keys=[]):
    #     for ens_name in ensembles:
    #         ens = self._repo().get_ensemble(ens_name)

    def get_param_data(self, ensembles, keys=[], realizations=[]):

        for ens_name in ensembles:
            ens = self._repo().get_ensemble(ens_name)

            yield DataStreamMessage("ensemble", ens_name, "start")

            for param_def in ens.parameter_definitions:
                if param_def.name in keys or len(keys) == 0:
                    size = 1
                    yield DataStreamMessage("key_def", param_def.name, "size", size)

            for response_def in ens.response_definitions:
                if response_def.name in keys or len(keys) == 0:
                    size = len(response_def.responses[0].values)
                    yield DataStreamMessage("key_def", response_def.name, "size", size)

            for param_def in ens.parameter_definitions:
                if param_def.name in keys or len(keys) == 0:
                    yield DataStreamMessage("index_def", param_def.name, "indexes", [0])

            for response_def in ens.response_definitions:
                if response_def.name in keys or len(keys) == 0:
                    yield DataStreamMessage("index_def", response_def.name, "indexes", response_def.indexes)

            for real in ens.realizations:
                if real.id in realizations or len(realizations) == 0:
                    yield DataStreamMessage("realisation", real.id, "start")
                    for param_def in ens.parameter_definitions:
                        if param_def.name in keys or len(keys) == 0:
                            for param in real.parameters:
                                if param.parameter_definition == param_def:
                                    yield DataStreamMessage("parameter", param_def.name, "start")
                                    yield DataStreamMessage("parameter", param_def.name, "chunk", [param.value])
                            yield DataStreamMessage("parameter", param_def.name, "done")

                    for response_def in ens.response_definitions:
                        if response_def.name in keys or len(keys) == 0:
                            for respons in real.responses:
                                if respons.response_definition == response_def:
                                    yield DataStreamMessage("response", response_def.name, "start")
                                    yield DataStreamMessage("response", response_def.name, "chunk", respons.values)
                            yield DataStreamMessage("response", response_def.name, "done")

                    yield DataStreamMessage("realisation", real.id, "done")

            yield DataStreamMessage("ensemble", ens_name, "stop")

        yield DataStreamMessage("EOL", "EOL", "EOL")

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

class DataStreamMessage(object):

    def __init__(self, type, name, status, value=None):
        self.value = value
        self.status = status
        self.name = name
        self.type = type

    def __repr__(self):
        return "DataStreamMessage: " + str(self.__dict__)