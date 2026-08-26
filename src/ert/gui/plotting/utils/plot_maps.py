from collections.abc import Callable

from ert.config.known_response_types import KNOWN_ERT_RESPONSE_TYPES
from ert.gui.plotting.ert_plots import (
    CrossEnsembleStatisticsPlot,
    DistributionPlot,
    GaussianKDEPlot,
    HistogramPlot,
    MisfitsPlot,
    StatisticsPlot,
    StdDevPlot,
)
from ert.gui.plotting.everest_plots import (
    EverestBatchObjectiveFunctionPlot,
    EverestConstraintsPlot,
    EverestControlsPlot,
    EverestGradientsPlot,
    EverestObjectiveFunctionPlot,
)
from ert.gui.plotting.shared_plots.ensemble import EnsemblePlot
from ert.gui.plotting.widgets.plot_widget import Plotter

CROSS_ENSEMBLE_STATISTICS = "Cross ensemble statistics"
DISTRIBUTION = "Distribution"
GAUSSIAN_KDE = "Gaussian KDE"
ENSEMBLE = "Ensemble"
HISTOGRAM = "Histogram"
STATISTICS = "Statistics"
STD_DEV = "Std dev"
MISFITS = "Misfits"
EVEREST_CONTROLS_PLOT = "Controls"
EVEREST_GRADIENTS_PLOT = "Gradient"
EVEREST_OBJECTIVE_FUNCTION_PLOT = "Objective function"
EVEREST_BATCH_OBJECTIVE_FUNCTION_PLOT = "Total objective value"
EVEREST_CONSTRAINT_PLOT = "Constraints"

ERT_PLOT_MAP: dict[str, Callable[[], Plotter]] = {
    STATISTICS: StatisticsPlot,
    MISFITS: MisfitsPlot,
    HISTOGRAM: HistogramPlot,
    GAUSSIAN_KDE: GaussianKDEPlot,
    DISTRIBUTION: DistributionPlot,
    CROSS_ENSEMBLE_STATISTICS: CrossEnsembleStatisticsPlot,
    STD_DEV: StdDevPlot,
}
EVEREST_PLOT_MAP: dict[str, Callable[[], Plotter]] = {
    EVEREST_BATCH_OBJECTIVE_FUNCTION_PLOT: EverestBatchObjectiveFunctionPlot,
    EVEREST_OBJECTIVE_FUNCTION_PLOT: EverestObjectiveFunctionPlot,
    EVEREST_CONSTRAINT_PLOT: EverestConstraintsPlot,
    EVEREST_CONTROLS_PLOT: EverestControlsPlot,
    EVEREST_GRADIENTS_PLOT: EverestGradientsPlot,
}
SHARED_PLOT_MAP: dict[str, Callable[[], Plotter]] = {
    ENSEMBLE: EnsemblePlot,
}

ERT_RESPONSE_ORIGINS: frozenset[str] = frozenset(
    response_type.model_fields["type"].default
    for response_type in KNOWN_ERT_RESPONSE_TYPES
)

_ERT_RESPONSE_TABS = [ENSEMBLE, STATISTICS, MISFITS]

TABS_FOR_DATA_ORIGIN: dict[str, list[str]] = {
    "gen_kw": [HISTOGRAM, GAUSSIAN_KDE, DISTRIBUTION, CROSS_ENSEMBLE_STATISTICS],
    "surface": _ERT_RESPONSE_TABS,
    "field": [STD_DEV],
    "everest_parameters": [EVEREST_CONTROLS_PLOT],
    "everest_objectives": [
        EVEREST_OBJECTIVE_FUNCTION_PLOT,
        EVEREST_GRADIENTS_PLOT,
    ],
    "everest_constraints": [EVEREST_CONSTRAINT_PLOT, EVEREST_GRADIENTS_PLOT],
    "everest_batch_objectives": [EVEREST_BATCH_OBJECTIVE_FUNCTION_PLOT],
    **dict.fromkeys(ERT_RESPONSE_ORIGINS, _ERT_RESPONSE_TABS),
}
