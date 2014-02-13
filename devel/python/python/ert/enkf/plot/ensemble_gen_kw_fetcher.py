# Copyright (C) 2014  Statoil ASA, Norway.
#
# The file 'ensemble_gen_kw_fetcher.py' is part of ERT - Ensemble based Reservoir Tool.
#
# ERT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ERT is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
# for more details.
from ert.enkf import EnkfObservationImplementationType
from ert.enkf.data.gen_kw_config import GenKwConfig
from ert.enkf.enums.ert_impl_type_enum import ErtImplType
from ert.enkf.observations import BlockObservation
from ert.enkf.plot import DataFetcher


class EnsembleGenKWFetcher(DataFetcher):
    def __init__(self, ert):
        super(EnsembleGenKWFetcher, self).__init__(ert)

    def fetchSupportedKeys(self):
        gen_kw_keys = self.ert().ensembleConfig().getKeylistFromImplType(ErtImplType.GEN_KW)
        gen_kw_list = []
        for key in gen_kw_keys:
            enkf_config_node = self.ert().ensembleConfig().getNode(key)
            model_config = enkf_config_node.getModelConfig()
            assert isinstance(model_config, GenKwConfig)

            model_keys = model_config.getKeyWords()
            for key_word in model_keys:
                gen_kw_list.append("%s:%s" % (key, key_word))

        return gen_kw_list


    def fetchData(self, key, case=None):
        return None

