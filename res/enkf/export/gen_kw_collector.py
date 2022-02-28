from res import _lib
from ecl.util.util import BoolVector
from pandas import DataFrame
from res.enkf import EnKFMain
from res.enkf.enums import RealizationStateEnum
from res.enkf.key_manager import KeyManager


class GenKwCollector:
    @staticmethod
    def createActiveList(ert, fs):
        state_map = fs.getStateMap()
        ens_mask = state_map.selectMatching(
            RealizationStateEnum.STATE_INITIALIZED
            | RealizationStateEnum.STATE_HAS_DATA,
        )
        index_list = [index for index, element in enumerate(ens_mask) if element]
        bool_vec = BoolVector.createFromList(len(index_list), index_list)
        return bool_vec.createActiveList()

    @staticmethod
    def getAllGenKwKeys(ert):
        """@rtype: list of str"""
        key_manager = KeyManager(ert)
        return key_manager.genKwKeys()

    @staticmethod
    def loadAllGenKwData(ert: EnKFMain, case_name, keys=None, realization_index=None):
        """
        @type ert: EnKFMain
        @type case_name: str
        @type keys: list of str
        @rtype: DataFrame
        """
        fs = ert.getEnkfFsManager().getFileSystem(case_name)

        realizations = GenKwCollector.createActiveList(ert, fs)

        if realization_index is not None:
            if realization_index not in realizations:
                raise IndexError(f"No such realization ({realization_index})")
            realizations = [realization_index]

        gen_kw_keys = GenKwCollector.getAllGenKwKeys(ert)

        if keys is not None:
            gen_kw_keys = [
                key for key in keys if key in gen_kw_keys
            ]  # ignore keys that doesn't exist

        gen_kw_array = _lib.enkf_fs_keyword_data.keyword_data_get_realizations(
            ert.ensembleConfig(), fs, gen_kw_keys, realizations
        )
        gen_kw_data = DataFrame(
            data=gen_kw_array, index=realizations, columns=gen_kw_keys
        )

        gen_kw_data.index.name = "Realization"
        return gen_kw_data
