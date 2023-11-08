from ert.enum_shim import StrEnum


class AnalysisMode(StrEnum):
    ITERATED_ENSEMBLE_SMOOTHER = "IES_ENKF"
    ENSEMBLE_SMOOTHER = "STD_ENKF"
