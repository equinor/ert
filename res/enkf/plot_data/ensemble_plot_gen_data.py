#  Copyright (C) 2014 Equinor ASA, Norway.
#
#  The file 'ensemble_plot_gen_data.py' is part of ERT - Ensemble based Reservoir Tool.
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

from cwrap import BaseCClass
from ecl.util.util import DoubleVector
from res import ResPrototype
from res import _lib
from res.enkf.config import EnkfConfigNode
from res.enkf.enkf_fs import EnkfFs
from res.enkf.enums import ErtImplType
from res.enkf.plot_data.ensemble_plot_gen_data_vector import EnsemblePlotGenDataVector


class EnsemblePlotGenData(BaseCClass):
    TYPE_NAME = "ensemble_plot_gen_data"

    _alloc = ResPrototype("void* enkf_plot_gendata_alloc(enkf_config_node)", bind=False)
    _size = ResPrototype("int   enkf_plot_gendata_get_size(ensemble_plot_gen_data)")
    _load = ResPrototype(
        "void  enkf_plot_gendata_load"
        "(ensemble_plot_gen_data, enkf_fs, int, bool_vector)"
    )
    _get = ResPrototype(
        "ensemble_plot_gen_data_vector_ref enkf_plot_gendata_iget"
        "(ensemble_plot_gen_data, int)"
    )
    _min_values = ResPrototype(
        "double_vector_ref enkf_plot_gendata_get_min_values(ensemble_plot_gen_data)"
    )
    _max_values = ResPrototype(
        "double_vector_ref enkf_plot_gendata_get_max_values(ensemble_plot_gen_data)"
    )
    _free = ResPrototype("void  enkf_plot_gendata_free(ensemble_plot_gen_data)")

    def __init__(self, ensemble_config_node, file_system, report_step):
        assert isinstance(ensemble_config_node, EnkfConfigNode)
        assert ensemble_config_node.getImplementationType() == ErtImplType.GEN_DATA

        c_ptr = self._alloc(ensemble_config_node)
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError(
                "Unable to construct EnsemplePlotGenData from given config node!"
            )

        self.__load(file_system, report_step)

    def __load(self, file_system, report_step):
        assert isinstance(file_system, EnkfFs)

        self._load(file_system, report_step, None)

    def __len__(self):
        """@rtype: int"""
        return self._size()

    def __getitem__(self, index) -> EnsemblePlotGenDataVector:
        """@rtype: EnsemblePlotGenDataVector"""
        return self._get(index)

    def __iter__(self):
        cur = 0
        while cur < len(self):
            yield self[cur]
            cur += 1

    def getMaxValues(self) -> DoubleVector:
        """@rtype: DoubleVector"""
        return self._max_values().setParent(self)

    def getMinValues(self) -> DoubleVector:
        """@rtype: DoubleVector"""
        return self._min_values().setParent(self)

    def getRealizations(self, realizations):
        return _lib.enkf_fs_general_data.gendata_get_realizations(self, realizations)

    def free(self):
        self._free()

    def __repr__(self):
        return "EnsemblePlotGenData(size = %d) %s" % (len(self), self._ad_str())
