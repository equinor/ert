import numpy as np
from pandas import DataFrame, MultiIndex
from res import _lib
from res.enkf import EnKFMain
from res.enkf.enums import RealizationStateEnum
from res.enkf.enkf_fs import EnkfFs
from typing import List


class SummaryCollector:
    @staticmethod
    def getAllSummaryKeys(ert: EnKFMain) -> List[str]:
        return ert.getKeyManager().summaryKeys()

    @staticmethod
    def createActiveList(fs: EnkfFs) -> List[int]:
        state_map = fs.getStateMap()
        ens_mask = state_map.selectMatching(RealizationStateEnum.STATE_HAS_DATA)
        return [index for index, element in enumerate(ens_mask) if element]

    @staticmethod
    def loadAllSummaryData(
        ert: EnKFMain,
        case_name: str,
        keys: List[str] = None,
        realization_index: int = None,
    ) -> DataFrame:
        fs = ert.getEnkfFsManager().getFileSystem(case_name, read_only=True)

        time_map = fs.getTimeMap()
        dates = [time_map[index].datetime() for index in range(1, len(time_map))]

        realizations = SummaryCollector.createActiveList(fs)
        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization {realization_index}")
            realizations = [realization_index]

        summary_keys = ert.getKeyManager().summaryKeys()
        if keys is not None:
            summary_keys = [
                key for key in keys if key in summary_keys
            ]  # ignore keys that doesn't exist

        summary_data = _lib.enkf_fs_summary_data.get_summary_data(
            ert.ensembleConfig(), fs, summary_keys, realizations, len(dates)
        )

        if np.isnan(summary_data).all():
            return DataFrame()

        multi_index = MultiIndex.from_product(
            [realizations, dates], names=["Realization", "Date"]
        )

        return DataFrame(data=summary_data, index=multi_index, columns=summary_keys)
