from cwrap import BaseCClass
from ecl.util.util import BoolVector

from res import ResPrototype
from res.enkf.config import EnkfConfigNode
from res.enkf.enkf_fs import EnkfFs
from res.enkf.plot_data.ensemble_plot_data_vector import EnsemblePlotDataVector


class EnsemblePlotData(BaseCClass):
    TYPE_NAME = "ensemble_plot_data"

    _alloc = ResPrototype("void* enkf_plot_data_alloc(enkf_config_node)", bind=False)
    _load = ResPrototype(
        "void  enkf_plot_data_load(ensemble_plot_data, enkf_fs, char*, bool_vector)"
    )
    _size = ResPrototype("int   enkf_plot_data_get_size(ensemble_plot_data)")
    _get = ResPrototype(
        "ensemble_plot_data_vector_ref enkf_plot_data_iget(ensemble_plot_data, int)"
    )
    _free = ResPrototype("void  enkf_plot_data_free(ensemble_plot_data)")

    def __init__(
        self, ensemble_config_node, file_system=None, user_index=None, input_mask=None
    ):
        assert isinstance(ensemble_config_node, EnkfConfigNode)

        c_pointer = self._alloc(ensemble_config_node)
        super().__init__(c_pointer)

        if file_system is not None:
            self.load(file_system, user_index, input_mask)

    def load(self, file_system, user_index=None, input_mask=None):
        assert isinstance(file_system, EnkfFs)
        if input_mask is not None:
            assert isinstance(input_mask, BoolVector)

        self._load(file_system, user_index, input_mask)

    def __len__(self):
        """@rtype: int"""
        return self._size()

    def __getitem__(self, index) -> EnsemblePlotDataVector:
        """@rtype: EnsemblePlotDataVector"""
        return self._get(index)

    def __iter__(self):
        cur = 0
        while cur < len(self):
            yield self[cur]
            cur += 1

    def free(self):
        self._free()

    def __repr__(self):
        return "EnsemblePlotData(size = %d) %s" % (len(self), self._ad_str())
