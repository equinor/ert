class ErtWarning(Warning):
    """Base class for warnings in this module."""


class PostExperimentWarning(ErtWarning):
    """Warnings to be shown in GUI after the experiment has finished."""


class ObservationReportWarning(ErtWarning):
    """Warnings to be shown per-observation in the analysis report."""
