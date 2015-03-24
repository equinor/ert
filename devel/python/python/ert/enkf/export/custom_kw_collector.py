import math
from pandas import DataFrame
from ert.enkf import ErtImplType, EnKFMain, EnkfFs, RealizationStateEnum, CustomKWConfig, EnkfNode, NodeId, \
    EnkfStateType
from ert.enkf.plot_data import EnsemblePlotGenKW
from ert.util import BoolVector


class CustomKWCollector(object):

    @staticmethod
    def createActiveList(ert, fs):
        state_map = fs.getStateMap()
        ens_mask = BoolVector(False, ert.getEnsembleSize())
        state_map.selectMatching(ens_mask, RealizationStateEnum.STATE_HAS_DATA)
        active_list = BoolVector.createActiveList(ens_mask)

        return [iens for iens in active_list]

    @staticmethod
    def getAllCustomKWKeys(ert):
        """ @rtype: list of str """
        custom_kw_keys = ert.ensembleConfig().getKeylistFromImplType(ErtImplType.CUSTOM_KW)
        custom_kw_keys = [key for key in custom_kw_keys]

        custom_kw_list = []
        for name in custom_kw_keys:
            enkf_config_node = ert.ensembleConfig().getNode(name)
            custom_kw_config = enkf_config_node.getModelConfig()
            assert isinstance(custom_kw_config, CustomKWConfig)

            for key in custom_kw_config:
                custom_kw_list.append("%s:%s" % (name, key))

        return custom_kw_list

    @staticmethod
    def groupKeys(keys):
        grouped_keys = {}
        for key in keys:
            name, keyword = key.split(":")

            if not name in grouped_keys:
                grouped_keys[name] = []

            grouped_keys[name].append(keyword)

        return grouped_keys


    @staticmethod
    def loadAllCustomKWData(ert, case_name, keys=None):
        """
        @type ert: EnKFMain
        @type case_name: str
        @type keys: list of str
        @rtype: DataFrame
        """
        fs = ert.getEnkfFsManager().getFileSystem(case_name)

        realizations = fs.realizationList(RealizationStateEnum.STATE_HAS_DATA)

        custom_kw_keys = CustomKWCollector.getAllCustomKWKeys(ert)

        if keys is not None:
            custom_kw_keys = [key for key in keys if key in custom_kw_keys] # ignore keys that doesn't exist

        custom_kw_data = DataFrame(index=realizations, columns=custom_kw_keys)
        custom_kw_data.index.name = "Realization"

        custom_kw_keys = CustomKWCollector.groupKeys(custom_kw_keys)

        for name in custom_kw_keys:
            ensemble_config_node = ert.ensembleConfig().getNode(name)
            enkf_node = EnkfNode(ensemble_config_node)

            keys = custom_kw_keys[name]

            for realization_number in realizations:
                node_id = NodeId(0, realization_number, EnkfStateType.FORECAST)
                if enkf_node.tryLoad(fs, node_id):
                    custom_kw = enkf_node.asCustomKW()

                    for key in keys:
                        value = custom_kw[key]
                        custom_kw_data["%s:%s" % (name, key)][realization_number] = value

        return custom_kw_data

