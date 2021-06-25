from pandas import DataFrame, MultiIndex
import numpy
from res.enkf import (
    ErtImplType,
    EnKFMain,
    EnkfFs,
    RealizationStateEnum,
    EnkfObservationImplementationType,
)
from res.enkf.key_manager import KeyManager
from res.enkf.plot_data import EnsemblePlotData
from ecl.util.util import BoolVector


class SummaryObservationCollector(object):
    @staticmethod
    def getAllObservationKeys(ert):
        """
        @type ert: EnKFMain
        @rtype: list of str
        """
        key_manager = KeyManager(ert)
        return key_manager.summaryKeysWithObservations()

    @staticmethod
    def loadObservationData(ert, case_name, keys=None):
        """
        @type ert: EnKFMain
        @type case_name: str
        @type keys: list of str
        @rtype: DataFrame
        """
        observations = ert.getObservations()
        history_length = ert.getHistoryLength()
        dates = [
            observations.getObservationTime(index).datetime()
            for index in range(1, history_length + 1)
        ]
        summary_keys = SummaryObservationCollector.getAllObservationKeys(ert)
        if keys is not None:
            summary_keys = [
                key for key in keys if key in summary_keys
            ]  # ignore keys that doesn't exist
        columns = summary_keys
        std_columns = ["STD_%s" % key for key in summary_keys]
        df = DataFrame(index=dates, columns=columns + std_columns)
        for key in summary_keys:
            observation_keys = ert.ensembleConfig().getNode(key).getObservationKeys()
            for obs_key in observation_keys:
                observation_data = observations[obs_key]
                for index in range(0, history_length + 1):
                    if observation_data.isActive(index):
                        obs_time = observations.getObservationTime(index).datetime()
                        node = observation_data.getNode(index)
                        value = node.getValue()
                        std = node.getStandardDeviation()
                        df[key][obs_time] = value
                        df["STD_%s" % key][obs_time] = std
        return df

    @classmethod
    def summaryKeyHasObservations(cls, ert, key):
        return len(ert.ensembleConfig().getNode(key).getObservationKeys()) > 0
