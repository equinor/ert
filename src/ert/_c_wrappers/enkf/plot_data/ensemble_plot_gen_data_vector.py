from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype


class EnsemblePlotGenDataVector(BaseCClass):
    TYPE_NAME = "ensemble_plot_gen_data_vector"

    _size = ResPrototype(
        "int    enkf_plot_genvector_get_size(ensemble_plot_gen_data_vector)"
    )
    _get_value = ResPrototype(
        "double enkf_plot_genvector_iget(ensemble_plot_gen_data_vector, int)"
    )

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def __len__(self):
        """@rtype: int"""
        return self._size()

    def __repr__(self):
        return f"EnsemblePlotGenDataVector(size = {len(self)}) {self._ad_str()})"

    def getValue(self, index):
        """@rtype: float"""
        return self[index]

    def __iter__(self):
        cur = 0
        while cur < len(self):
            yield self[cur]
            cur += 1

    def __getitem__(self, index):
        """@rtype: float"""
        return self._get_value(index)
