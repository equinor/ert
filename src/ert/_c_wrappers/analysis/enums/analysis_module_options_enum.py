from cwrap import BaseCEnum


class AnalysisModuleOptionsEnum(BaseCEnum):
    TYPE_NAME = "analysis_module_options_enum"
    ANALYSIS_USE_A = None
    ANALYSIS_UPDATE_A = None
    ANALYSIS_ITERABLE = None


AnalysisModuleOptionsEnum.addEnum("ANALYSIS_USE_A", 4)
AnalysisModuleOptionsEnum.addEnum("ANALYSIS_UPDATE_A", 8)
AnalysisModuleOptionsEnum.addEnum("ANALYSIS_ITERABLE", 32)
