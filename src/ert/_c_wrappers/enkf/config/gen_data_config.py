from typing import List

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype


class GenDataConfig(BaseCClass):
    TYPE_NAME = "gen_data_config"

    _alloc = ResPrototype(
        "void* gen_data_config_alloc_GEN_DATA_result( char*)",  # noqa
        bind=False,
    )
    _free = ResPrototype("void  gen_data_config_free( gen_data_config )")
    _has_report_step = ResPrototype(
        "bool  gen_data_config_has_report_step(gen_data_config, int)"
    )
    _get_key = ResPrototype("char* gen_data_config_get_key(gen_data_config)")
    _get_num_report_step = ResPrototype(
        "int   gen_data_config_num_report_step(gen_data_config)"
    )
    _iget_report_step = ResPrototype(
        "int   gen_data_config_iget_report_step(gen_data_config, int)"
    )

    def __init__(self, key):
        # Can currently only create GEN_DATA instances which should be used
        # as result variables.
        c_pointer = self._alloc(key)
        super().__init__(c_pointer)

    def getName(self):
        return self.name()

    def name(self):
        return self._get_key()

    def free(self):
        self._free()

    def __repr__(self):
        return f"GenDataConfig(key={self.name()})"

    def hasReportStep(self, report_step) -> bool:
        return self._has_report_step(report_step)

    def getNumReportStep(self) -> int:
        return self._get_num_report_step()

    def getReportStep(self, index) -> int:
        return self._iget_report_step(index)

    def getReportSteps(self) -> List[int]:
        return [self.getReportStep(index) for index in range(self.getNumReportStep())]

    def __eq__(self, other) -> bool:
        if self.getName() != other.getName():
            return False

        if self.getReportSteps() != other.getReportSteps():
            return False

        return True
