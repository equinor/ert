from typing import List

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype


class GenDataConfig(BaseCClass):
    TYPE_NAME = "gen_data_config"

    _alloc = ResPrototype(
        "void* gen_data_config_alloc_GEN_DATA_result()",  # noqa
        bind=False,
    )
    _free = ResPrototype("void  gen_data_config_free( gen_data_config )")
    _has_report_step = ResPrototype(
        "bool  gen_data_config_has_report_step(gen_data_config, int)"
    )
    _get_num_report_step = ResPrototype(
        "int   gen_data_config_num_report_step(gen_data_config)"
    )
    _iget_report_step = ResPrototype(
        "int   gen_data_config_iget_report_step(gen_data_config, int)"
    )

    def __init__(self, key):
        c_pointer = self._alloc()
        self.key = key
        super().__init__(c_pointer)

    def getKey(self):
        return self.key

    def free(self):
        self._free()

    def __repr__(self):
        return f"GenDataConfig(key={self.key})"

    def hasReportStep(self, report_step) -> bool:
        return self._has_report_step(report_step)

    def getNumReportStep(self) -> int:
        return self._get_num_report_step()

    def getReportStep(self, index) -> int:
        return self._iget_report_step(index)

    def getReportSteps(self) -> List[int]:
        return [self.getReportStep(index) for index in range(self.getNumReportStep())]

    def __eq__(self, other) -> bool:
        if self.getKey() != other.getKey():
            return False

        if self.getReportSteps() != other.getReportSteps():
            return False

        return True
