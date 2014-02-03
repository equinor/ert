from ert.enkf.enums import ErtImplType, EnkfObservationImplementationType
from ert.enkf.plot.block_observation_data_fetcher import BlockObservationDataFetcher
from ert_gui.models import ErtConnector
from ert_gui.models.mixins.list_model import ListModelMixin


class DataTypeKeysModel(ErtConnector, ListModelMixin):

    def __init__(self):
        self.__keys = None
        self.__observation_keys = None
        self.__block_observation_keys = None
        self.__summary_keys = None
        super(DataTypeKeysModel, self).__init__()


    def getAllKeys(self):
        """ @rtype: list of str """
        keys = self.getAllBlockObservationKeys() + self.getAllSummaryKeys()
        return sorted([key for key in keys], key=lambda k : k.lower())

    def getAllSummaryKeys(self):
        """ @rtype: list of str """
        if self.__summary_keys is None:
            keys = self.ert().ensembleConfig().getKeylistFromImplType(ErtImplType.SUMMARY)
            self.__summary_keys = sorted([key for key in keys], key=lambda k : k.lower())

        return self.__summary_keys

    def getAllBlockObservationKeys(self):
        """ @rtype: list of str """
        if self.__block_observation_keys is None:
            self.__block_observation_keys = BlockObservationDataFetcher(self.ert()).getSupportedKeys()

        return self.__block_observation_keys

    def getAllObservationSummaryKeys(self):
        """ @rtype: list of str """
        if self.__observation_keys is None:
            self.__observation_keys = [key for key in self.getAllSummaryKeys() if self.__isSummaryKeyObservationKey(key)]
        return self.__observation_keys


    def getList(self):
        """ @rtype: list of str """
        self.__keys = self.__keys or self.getAllKeys()
        return self.__keys


    def isObservationKey(self, item):
        """ @rtype: bool """
        if self.__observation_keys is None:
            self.getAllObservationSummaryKeys()

        if self.__block_observation_keys is None:
            self.getAllBlockObservationKeys()

        return item in self.__observation_keys or item in self.__block_observation_keys

    def __isSummaryKeyObservationKey(self, key):
        return len(self.ert().ensembleConfig().getNode(key).getObservationKeys()) > 0

    def isSummaryKey(self, key):
        return key in self.__summary_keys

    def isBlockKey(self, key):
        return key in self.__block_observation_keys

