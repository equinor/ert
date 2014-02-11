#  Copyright (C) 2014 Statoil ASA, Norway.
#
#  The file 'ensemble_plot_gen_kw.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.

from ert.cwrap import BaseCClass, CWrapper
from ert.enkf import ENKF_LIB
from ert.enkf.data import EnkfConfigNode
from ert.enkf.enkf_fs import EnkfFs
from ert.enkf.enums.enkf_state_type_enum import EnkfStateType
from ert.enkf.enums.ert_impl_type_enum import ErtImplType
from ert.util import BoolVector


class EnsemblePlotGenKW(BaseCClass):
    def __init__(self, ensemble_config_node, file_system, keyword, state=EnkfStateType.FORECAST, input_mask=None):
        assert isinstance(ensemble_config_node, EnkfConfigNode)
        assert ensemble_config_node.getImplementationType() == ErtImplType.GEN_KW

        c_pointer = EnsemblePlotGenKW.cNamespace().alloc(ensemble_config_node)
        super(EnsemblePlotGenKW, self).__init__(c_pointer)

        self.__load(file_system, keyword, state, input_mask)


    def __load(self, file_system, keyword, state=EnkfStateType.FORECAST, input_mask=None):
        assert isinstance(file_system, EnkfFs)
        assert isinstance(state, EnkfStateType)
        if not input_mask is None:
            assert isinstance(input_mask, BoolVector)
        transform = True
        EnsemblePlotGenKW.cNamespace().load(self, file_system, transform, 0, state, input_mask)

    def __len__(self):
        """ @rtype: int """
        return EnsemblePlotGenKW.cNamespace().size(self)

    def __getitem__(self, index):
        """ @rtype: EnsemblePlotGenDataVector """
        return EnsemblePlotGenKW.cNamespace().get(self, index)

    def __iter__(self):
        cur = 0
        while cur < len(self):
            yield self[cur]
            cur += 1



    def free(self):
        EnsemblePlotGenKW.cNamespace().free(self)



cwrapper = CWrapper(ENKF_LIB)
cwrapper.registerType("ensemble_plot_gen_kw", EnsemblePlotGenKW)
cwrapper.registerType("ensemble_plot_gen_kw_obj", EnsemblePlotGenKW.createPythonObject)
cwrapper.registerType("ensemble_plot_gen_kw_ref", EnsemblePlotGenKW.createCReference)

EnsemblePlotGenKW.cNamespace().free = cwrapper.prototype("void enkf_plot_gen_kw_free(ensemble_plot_gen_kw)")
EnsemblePlotGenKW.cNamespace().alloc = cwrapper.prototype("c_void_p enkf_plot_gen_kw_alloc(enkf_config_node)")

EnsemblePlotGenKW.cNamespace().size = cwrapper.prototype("int enkf_plot_gen_kw_get_size(ensemble_plot_gen_kw)")
EnsemblePlotGenKW.cNamespace().load = cwrapper.prototype("void enkf_plot_gen_kw_load(ensemble_plot_gen_kw, enkf_fs, bool, int, enkf_state_type_enum, bool_vector)")
EnsemblePlotGenKW.cNamespace().get = cwrapper.prototype("ensemble_plot_gen_kw_vector_ref enkf_plot_gen_kw_iget(ensemble_plot_gen_kw, int)")







