from ert.enkf import ErtImplType


class KeyManager(object):

    def __init__(self, ert):
        super(KeyManager, self).__init__()
        """
        @type ert: ert.enkf.EnKFMain
        """
        self.__ert = ert

        self.__all_keys = None
        self.__all_keys_with_observations = None
        self.__summary_keys = None
        self.__summary_keys_with_observations = None


    def ert(self):
        """ :rtype:  ert.enkf.EnKFMain """
        return self.__ert

    def ensembleConfig(self):
        """ :rtype: ert.enkf.EnsembleConfig """
        return self.ert().ensembleConfig()

    def summaryKeys(self):
        """ :rtype: list of Str """
        if self.__summary_keys is None:
            self.__summary_keys = sorted([key for key in self.ensembleConfig().getKeylistFromImplType(ErtImplType.SUMMARY)], key=lambda k : k.lower())

        return self.__summary_keys

    def summaryKeysWithObservations(self):
        """ :rtype: list of Str """
        if self.__summary_keys_with_observations is None:
            self.__summary_keys_with_observations = sorted([key for key in self.summaryKeys() if len(self.ensembleConfig().getNode(key).getObservationKeys()) > 0], key=lambda k : k.lower())

        return self.__summary_keys_with_observations

    def allDataTypeKeys(self):
        """ :rtype: list of Str """
        if self.__all_keys is None:
            self.__all_keys = self.summaryKeys()

        return self.__all_keys

    def allDataTypeKeysWithObservations(self):
        """ :rtype: list of Str """
        if self.__all_keys_with_observations is None:
            self.__all_keys_with_observations = self.summaryKeysWithObservations()

        return self.__all_keys_with_observations

    def isKeyWithObservations(self, key):
        """ :rtype: bool """
        return key in self.allDataTypeKeysWithObservations()

    def isSummaryKey(self, key):
        """ :rtype: bool """
        return key in self.summaryKeys()
