from cwrap import BaseCEnum


class AnalysisModeEnum(BaseCEnum):
    TYPE_NAME = "analysis_mode_enum"
    ENSEMBLE_SMOOTHER = None
    ITERATED_ENSEMBLE_SMOOTHER = None


AnalysisModeEnum.addEnum("ENSEMBLE_SMOOTHER", 1)
AnalysisModeEnum.addEnum("ITERATED_ENSEMBLE_SMOOTHER", 2)
