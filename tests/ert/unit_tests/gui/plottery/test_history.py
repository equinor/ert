import pandas as pd
import pytest
from matplotlib.figure import Figure

from ert.gui.tools.plot.plottery.plot_config import PlotConfig
from ert.gui.tools.plot.plottery.plot_context import PlotContext
from ert.gui.tools.plot.plottery.plots.history import plot_history

HISTORY_DATA_VALUES = [10, 11, 12, 13, 14]


@pytest.fixture
def generic_plot_context():
    plot_config = PlotConfig()

    return PlotContext(
        plot_config,
        ensembles=[],
        ensembles_color_indexes=[],
        key="test",
    )


def history_dataframe():
    return pd.DataFrame({"history": HISTORY_DATA_VALUES})


@pytest.fixture
def plot_context_with_history(generic_plot_context):
    generic_plot_context.plotConfig().set_history_enabled(True)
    generic_plot_context.history_data = history_dataframe()
    return generic_plot_context


@pytest.fixture
def axes():
    return Figure().add_subplot(111)


@pytest.mark.parametrize(
    ("history_enabled", "data"),
    [(False, history_dataframe()), (True, None), (True, pd.DataFrame())],
)
def test_that_history_plot_returns_early_if_history_is_disabled_or_lacking_data(
    generic_plot_context, axes, history_enabled, data
):
    generic_plot_context.plotConfig().set_history_enabled(history_enabled)
    generic_plot_context.history_data = data

    plot_history(generic_plot_context, axes)
    axes_lines = axes.get_lines()
    assert len(axes_lines) == 0


def test_that_plot_history_returns_lines_if_history_is_enabled(
    plot_context_with_history, axes
):
    plot_history(plot_context_with_history, axes)
    lines = axes.get_lines()
    assert len(lines) == 1
    line = lines[0]
    assert list(line.get_xdata()) == list(range(len(HISTORY_DATA_VALUES)))
    assert list(line.get_ydata()) == HISTORY_DATA_VALUES


# visible just means self.line_style or self.marker is set
def test_that_legend_item_is_added_if_history_is_visible(
    plot_context_with_history, axes
):
    plot_history(plot_context_with_history, axes)
    plot_config = plot_context_with_history.plotConfig()
    assert "History" in plot_config.legend_labels()
    assert len(plot_config.legend_items()) == 1


def test_that_history_style_is_applied_initially(plot_context_with_history, axes):
    plot_config = plot_context_with_history.plotConfig()
    style = plot_config.history_style()

    plot_history(plot_context_with_history, axes)
    assert len(axes.get_lines()) == 1
    line = axes.get_lines()[0]
    assert line.get_color() == style.color
    assert line.get_alpha() == style.alpha
    assert line.get_linewidth() == style.width
    assert line.get_marker() == style.marker
    assert line.get_markersize() == style.size
    assert line.get_linestyle() == style.line_style
