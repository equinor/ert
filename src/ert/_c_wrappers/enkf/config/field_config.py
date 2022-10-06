from cwrap import BaseCClass
from ecl.grid import EclGrid

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.enums import EnkfFieldFileFormatEnum

from .field_type_enum import FieldTypeEnum


class FieldConfig(BaseCClass):
    TYPE_NAME = "field_config"

    _alloc = ResPrototype(
        "void*  field_config_alloc_empty(char* , ecl_grid , void* , bool)", bind=False
    )
    _free = ResPrototype("void   field_config_free( field_config )")
    _get_type = ResPrototype("field_type_enum field_config_get_type(field_config)")
    _get_truncation_mode = ResPrototype(
        "int    field_config_get_truncation_mode(field_config)"
    )
    _get_truncation_min = ResPrototype(
        "double field_config_get_truncation_min(field_config)"
    )
    _get_truncation_max = ResPrototype(
        "double field_config_get_truncation_max(field_config)"
    )
    _get_init_transform_name = ResPrototype(
        "char*  field_config_get_init_transform_name(field_config)"
    )
    _get_output_transform_name = ResPrototype(
        "char*  field_config_get_output_transform_name(field_config)"
    )
    _ijk_active = ResPrototype(
        "bool   field_config_ijk_active(field_config, int, int, int)"
    )
    _get_nx = ResPrototype("int    field_config_get_nx(field_config)")
    _get_ny = ResPrototype("int    field_config_get_ny(field_config)")
    _get_nz = ResPrototype("int    field_config_get_nz(field_config)")
    _get_grid = ResPrototype("ecl_grid_ref field_config_get_grid(field_config)")
    _get_data_size = ResPrototype(
        "int field_config_get_data_size_from_grid(field_config)"
    )
    _export_format = ResPrototype(
        "enkf_field_file_format_enum field_config_default_export_format(char*)",
        bind=False,
    )
    _guess_filetype = ResPrototype(
        "enkf_field_file_format_enum field_config_guess_file_type(char*)", bind=False
    )

    def __init__(self, kw, grid):
        c_ptr = self._alloc(kw, grid, None, False)
        super().__init__(c_ptr)

    @classmethod
    def exportFormat(cls, filename):
        export_format = cls._export_format(filename)
        if export_format in [
            EnkfFieldFileFormatEnum.ECL_GRDECL_FILE,
            EnkfFieldFileFormatEnum.RMS_ROFF_FILE,
        ]:
            return export_format
        else:
            raise ValueError(
                f"Could not determine grdecl / roff format from:{filename}"
            )

    @classmethod
    def guessFiletype(cls, filename):
        return cls._guess_filetype(filename)

    def get_type(self) -> FieldTypeEnum:
        return self._get_type()

    def get_truncation_mode(self):
        return self._get_truncation_mode()

    def get_truncation_min(self):
        return self._get_truncation_min()

    def get_init_transform_name(self):
        return self._get_init_transform_name()

    def get_output_transform_name(self):
        return self._get_output_transform_name()

    def get_truncation_max(self):
        return self._get_truncation_max()

    def get_nx(self):
        return self._get_nx()

    def get_ny(self):
        return self._get_ny()

    def get_nz(self):
        return self._get_nz()

    def get_data_size(self) -> int:
        return self._get_data_size()

    def get_grid(self) -> EclGrid:
        return self._get_grid()

    def ijk_active(self, i, j, k):
        return self._ijk_active(i, j, k)

    def free(self):
        self._free()

    def __repr__(self):
        return self._create_repr(
            f"type = {self.get_type()}, "
            f"nx = {self.get_nx()}, ny = {self.get_ny()}, nz = {self.get_nz()}"
        )

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        """@rtype: bool"""
        if self.get_init_transform_name() != other.get_init_transform_name():
            return False
        if self.get_output_transform_name() != other.get_output_transform_name():
            return False
        if self.get_truncation_max() != other.get_truncation_max():
            return False
        if self.get_truncation_min() != other.get_truncation_min():
            return False
        if self.get_truncation_mode() != other.get_truncation_mode():
            return False
        if self.get_type() != other.get_type():
            return False

        return True
