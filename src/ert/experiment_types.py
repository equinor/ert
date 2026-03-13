from enum import StrEnum


class ExperimentType(StrEnum):
    UNDEFINED = "Undefined"
    SINGLE_TEST_RUN = "Single Test Run"
    ENSEMBLE_EXPERIMENT = "Ensemble Experiment"
    EVALUATE_ENSEMBLE = "Evaluate Ensemble"
    ES_MDA = "Multiple Data Assimilation"
    ENSEMBLE_SMOOTHER = "Ensemble Smoother"
    ENSEMBLE_INFORMATION_FILTER = "Ensemble Information Filter"
    MANUAL_UPDATE = "Manual Update"
    MANUAL = "Manual"
    EVEREST = "Everest"
