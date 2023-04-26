from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype


class SummaryConfig(BaseCClass):
    TYPE_NAME = "summary_config"
    _alloc = ResPrototype("void* summary_config_alloc(char*)", bind=False)
    _free = ResPrototype("void  summary_config_free(summary_config)")

    def __init__(self, key):
        c_ptr = self._alloc(key)
        super().__init__(c_ptr)

    def __repr__(self):
        return f"SummaryConfig() {self._ad_str()}"

    def free(self):
        self._free()
