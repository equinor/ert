from pandas import DataFrame
from res.enkf import EnKFMain
from typing import List


class SummaryObservationCollector:
    @staticmethod
    def getAllObservationKeys(ert: EnKFMain) -> List[str]:
        return ert.getKeyManager().summaryKeysWithObservations()

    @staticmethod
    def loadObservationData(
        ert: EnKFMain, case_name: str, keys: List[str] = None
    ) -> DataFrame:
        observations = ert.getObservations()
        history_length = ert.getHistoryLength()
        dates = [
            observations.getObservationTime(index).datetime()
            for index in range(1, history_length + 1)
        ]
        summary_keys = ert.getKeyManager().summaryKeysWithObservations()
        if keys is not None:
            summary_keys = [
                key for key in keys if key in summary_keys
            ]  # ignore keys that doesn't exist
        columns = summary_keys
        std_columns = [f"STD_{key}" for key in summary_keys]
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
                        df[f"STD_{key}"][obs_time] = std
        return df

    @classmethod
    def summaryKeyHasObservations(cls, ert: EnKFMain, key: str) -> bool:
        return len(ert.ensembleConfig().getNode(key).getObservationKeys()) > 0
