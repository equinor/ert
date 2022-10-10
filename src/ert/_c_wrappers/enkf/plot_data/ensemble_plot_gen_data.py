from typing import List

from cwrap import BaseCClass
from ecl.util.util import DoubleVector

from ert import _clib
from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.config import EnkfConfigNode
from ert._c_wrappers.enkf.enkf_fs import EnkfFs
from ert._c_wrappers.enkf.enums import ErtImplType
from ert._c_wrappers.enkf.plot_data.ensemble_plot_gen_data_vector import (
    EnsemblePlotGenDataVector,
)


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

    def __init__(
        self,
        ensemble_config_node: EnkfConfigNode,
        file_system: EnkfFs,
        report_step: int,
    ):
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
        return self._max_values().setParent(self)

    def getMinValues(self) -> DoubleVector:
        return self._min_values().setParent(self)

    def getRealizations(self, realizations: List[int]):
        return _clib.enkf_fs_general_data.gendata_get_realizations(self, realizations)

    def free(self):
        self._free()

    def __repr__(self):
        return f"EnsemblePlotGenData(size = {len(self)}) {self._ad_str()}"
