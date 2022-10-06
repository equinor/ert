from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.enkf.enums import LoadFailTypeEnum


class SummaryConfig(BaseCClass):
    TYPE_NAME = "summary_config"
    _alloc = ResPrototype(
        "void* summary_config_alloc(char*, load_fail_type)", bind=False
    )
    _free = ResPrototype("void  summary_config_free(summary_config)")
    _get_var = ResPrototype("char* summary_config_get_var(summary_config)")

    def __init__(self, key, load_fail=LoadFailTypeEnum.LOAD_FAIL_WARN):
        c_ptr = self._alloc(key, load_fail)
        super().__init__(c_ptr)

    def __repr__(self):
        return f"SummaryConfig() {self._ad_str()}"

    def free(self):
        self._free()

    @property
    def key(self):
        return self._get_var()

    def __ne__(self, other):
        return not self == other

    def __eq__(self, other):
        """@rtype: bool"""
        if self.key != other.key:
            return False

        return True
