from collections.abc import Callable

from ert.gui.plotting.ert_plots import (
    CrossEnsembleStatisticsPlot,
    DistributionPlot,
    GaussianKDEPlot,
    HistogramPlot,
    MisfitMapPlot,
    MisfitsPlot,
    ObservationsMapPlot,
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
MISFIT_MAP = "Misfit map"
STATISTICS = "Statistics"
STD_DEV = "Std dev"
OBSERVATIONS_MAP = "Observations map"
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
    MISFIT_MAP: MisfitMapPlot,
    OBSERVATIONS_MAP: ObservationsMapPlot,
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
