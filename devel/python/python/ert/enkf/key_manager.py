from ert.enkf import ErtImplType, GenKwConfig


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
        self.__gen_kw_keys = None


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

    def genKwKeys(self):
        """ :rtype: list of Str """
        if self.__gen_kw_keys is None:
            gen_kw_keys = self.ert().ensembleConfig().getKeylistFromImplType(ErtImplType.GEN_KW)
            gen_kw_keys = [key for key in gen_kw_keys]

            gen_kw_list = []
            for key in gen_kw_keys:
                enkf_config_node = self.ert().ensembleConfig().getNode(key)
                gen_kw_config = enkf_config_node.getModelConfig()
                assert isinstance(gen_kw_config, GenKwConfig)

                for keyword_index, keyword in enumerate(gen_kw_config):
                    gen_kw_list.append("%s:%s" % (key, keyword))

                    if gen_kw_config.shouldUseLogScale(keyword_index):
                        gen_kw_list.append("LOG10_%s:%s" % (key, keyword))

            self.__gen_kw_keys = sorted(gen_kw_list, key=lambda k : k.lower())

        return self.__gen_kw_keys


    def allDataTypeKeys(self):
        """ :rtype: list of Str """
        if self.__all_keys is None:
            self.__all_keys = self.summaryKeys() + self.genKwKeys()

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

    def isGenKwKey(self, key):
        """ :rtype: bool """
        return key in self.genKwKeys()
