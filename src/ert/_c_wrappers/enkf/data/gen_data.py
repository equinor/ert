from cwrap import BaseCClass
from ecl.util.util import DoubleVector

from ert._c_wrappers import ResPrototype


class GenData(BaseCClass):
    TYPE_NAME = "gen_data"
    _alloc = ResPrototype("void*  gen_data_alloc()", bind=False)
    _free = ResPrototype("void   gen_data_free(gen_data)")
    _size = ResPrototype("int    gen_data_get_size(gen_data)")
    _iget = ResPrototype("double gen_data_iget_double(gen_data , int)")
    _export_data = ResPrototype("void   gen_data_export_data(gen_data , double_vector)")

    def __init__(self):
        c_ptr = self._alloc()
        if c_ptr:
            super().__init__(c_ptr)
        else:
            raise ValueError("Unable to construct GenData object.")

    def __len__(self) -> int:
        return self._size()

    def free(self):
        self._free()

    def __repr__(self):
        return f"GenData(len = {len(self)}) {self._ad_str()}"

    def getData(self) -> DoubleVector:
        data = DoubleVector()
        self._export_data(data)
        return data

    def __getitem__(self, idx):
        """Returns an item, or a list if idx is a slice.
        Note: When idx is a slice it does not return a new GenData!
        """
        ls = len(self)
        if isinstance(idx, int):
            if idx < 0:
                idx += ls
            if 0 <= idx < ls:
                return self._iget(idx)
            raise IndexError("List index out of range.")
        if isinstance(idx, slice):
            vec = self.getData()
            return [vec[i] for i in range(*idx.indices(ls))]
        raise TypeError(f"List indices must be integers, not {type(idx)}.")
