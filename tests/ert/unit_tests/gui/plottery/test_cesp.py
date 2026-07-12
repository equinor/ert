import pandas as pd
import pytest
from matplotlib.figure import Figure

from ert.gui.plotting.ert_plots import CrossEnsembleStatisticsPlot
from ert.gui.plotting.plot_api import EnsembleObject
from ert.gui.plotting.utils import PlotConfig, PlotContext, PlotStyle

STATISTIC_TYPES = ["mean", "p50", "min-max", "p10-p90", "std", "p33-p67"]
LEGEND_ITEMS = ["Mean", "P50", "Min/Max", "P10-P90", "Std dev", "P33-P67"]


@pytest.fixture
def ensemble_objects():
    return [
        EnsembleObject(
            f"iter_{i}",
            "id",
            False,
            "experiment_1",
            started_at="2012-12-10T00:00:00",
        )
        for i in range(3)
    ]


@pytest.fixture
def plot_context(ensemble_objects):
    return PlotContext(
        PlotConfig(),
        ensembles=ensemble_objects,
        ensembles_color_indexes=list(range(len(ensemble_objects))),
        key="test",
    )


@pytest.fixture
def ensemble_to_data(ensemble_objects):
    return {
        ensemble: pd.DataFrame({0: [1.0 + i, 2.0 + i, 3.0 + i]})
        for i, ensemble in enumerate(ensemble_objects)
    }


@pytest.mark.parametrize(
    ("connection_lines_enabled", "expected_lines"),
    [
        (
            True,
            5,
        ),  # 3 lines for mean, 1 line connecting each mean point (3 points -> 2 lines)
        (False, 3),  # Only 3 lines for mean
    ],
)
def test_that_connection_lines_is_plotted_when_enabled(
    plot_context, ensemble_to_data, connection_lines_enabled, expected_lines
):
    figure = Figure()
    config = plot_context.plotConfig()
    config.set_statistics_style("mean", PlotStyle("Mean", line_style="", marker="o"))
    config.set_distribution_line_enabled(connection_lines_enabled)

    CrossEnsembleStatisticsPlot().plot(
        figure,
        plot_context,
        ensemble_to_data,
        observation_data=pd.DataFrame(),
        std_dev_images={},
        obs_loc=None,
        key_def=None,
    )

    axes = figure.get_axes()[0]
    lines = axes.get_lines()
    assert len(lines) == expected_lines


def test_that_legend_displays_all_statistics_when_enabled(
    plot_context, ensemble_to_data
):
    figure = Figure()
    config = plot_context.plotConfig()
    for stat in STATISTIC_TYPES:
        config.set_statistics_style(
            stat, PlotStyle(stat.capitalize(), line_style="-", marker="o")
        )
    config.set_legend_enabled(True)

    CrossEnsembleStatisticsPlot().plot(
        figure,
        plot_context,
        ensemble_to_data,
        observation_data=pd.DataFrame(),
        std_dev_images={},
        obs_loc=None,
        key_def=None,
    )

    legend = figure.get_axes()[0].get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    for items in LEGEND_ITEMS:
        assert items in legend_texts


@pytest.mark.parametrize(
    ("statistic_type", "statistic_style", "expected_linestyle", "expected_marker"),
    [
        (stat_type, PlotStyle(label), "None", "")
        # line_style gets set to "" in plot_config
        # Marker gets set to "" if not provided in PlotStyle
        # (not provided in the _statistic_style PlotConfig)
        for stat_type, label in zip(STATISTIC_TYPES, LEGEND_ITEMS, strict=True)
    ],
)
def test_that_correct_default_statistic_style_is_applied(
    plot_context,
    ensemble_to_data,
    statistic_type,
    statistic_style,
    expected_linestyle,
    expected_marker,
):
    figure = Figure()
    config = plot_context.plotConfig()
    config.set_statistics_style(statistic_type, statistic_style)

    CrossEnsembleStatisticsPlot().plot(
        figure,
        plot_context,
        ensemble_to_data,
        observation_data=pd.DataFrame(),
        std_dev_images={},
        obs_loc=None,
        key_def=None,
    )

    lines = figure.get_axes()[0].get_lines()
    assert len(lines) > 0
    for line in lines:
        assert line.get_linestyle() == expected_linestyle
        assert line.get_marker() == expected_marker
