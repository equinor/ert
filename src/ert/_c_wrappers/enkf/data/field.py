import sys

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.config import FieldConfig


class Field(BaseCClass):
    TYPE_NAME = "field"

    _free = ResPrototype("void field_free( field )")
    _get_size = ResPrototype("int field_get_size(field)")
    _ijk_get_double = ResPrototype("double field_ijk_get_double(field, int, int, int)")
    _iget_double = ResPrototype("double field_iget_double(field, int)")
    _export = ResPrototype(
        "void field_export"
        "(field, char* , fortio , enkf_field_file_format_enum , bool , char*)"
    )

    def __init__(self):
        raise NotImplementedError("Class can not be instantiated directly!")

    def __len__(self):
        return self._get_size()

    def ijk_get_double(self, i, j, k):
        return self._ijk_get_double(i, j, k)

    def __getitem__(self, index):
        if 0 <= index < len(self):
            return self._iget_double(index)
        else:
            raise IndexError(f"Index: {index} out of range: [0,{len(self)})")

    def export(self, filename, file_type=None, init_file=None):
        output_transform = False
        if file_type is None:
            try:
                file_type = FieldConfig.exportFormat(filename)
            except ValueError:
                sys.stderr.write(
                    f"Sorry - could not infer output format from filename:{filename}\n"
                )
                return False

        self._export(filename, None, file_type, output_transform, init_file)
        return True

    def free(self):
        self._free()
