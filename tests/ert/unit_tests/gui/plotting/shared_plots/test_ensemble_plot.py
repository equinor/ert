from unittest.mock import Mock, patch

import pandas as pd
import pytest
from matplotlib.figure import Figure
from resfo_utilities import is_rate

from ert.gui.plotting.plot_api import EnsembleObject
from ert.gui.plotting.shared_plots import EnsemblePlot
from ert.gui.plotting.utils import PlotConfig, PlotContext
from ert.gui.plotting.widgets.plot_ensemble_selection_widget import (
    EnsembleSelectListWidget,
)


@pytest.fixture(
    params=[
        pytest.param("WOPR:OP_4"),
        pytest.param("BPR:123"),
    ]
)
def plot_context(request):
    context = Mock(spec=PlotContext)
    context.ensembles.return_value = [
        EnsembleObject(
            "ensemble_1", "id", False, "experiment_1", started_at="2012-12-10T00:00:00"
        )
    ]
    context.ensembles_color_indexes.return_value = [1]
    context.key.return_value = request.param
    context.history_data = None
    context.plotConfig.return_value = PlotConfig(title="Ensemble Plot")
    return context


def test_ensemble_plot_handles_rate(plot_context: PlotContext):
    figure = Figure()
    with patch(
        "ert.gui.plotting.shared_plots.ensemble.EnsemblePlot._plotLines"
    ) as mock_plotLines:
        EnsemblePlot().plot(
            figure,
            plot_context,
            dict.fromkeys(
                plot_context.ensembles(),
                pd.DataFrame([[0.1], [0.2], [0.3], [0.4], [0.5]]),
            ),
            pd.DataFrame(),
            {},
            {},
        )
        if is_rate(plot_context.key()):
            assert mock_plotLines.call_args[0][4] == "steps-pre"
        else:
            assert mock_plotLines.call_args[0][4] is None


def test_plot_ensemble_selection_widget_sorts_input_ensembles_reverse_chronologically(
    qtbot,
):
    times = [
        "2001-11-11T00:00:00",  # Second
        "2000-00-00T00:00:00",  # First
        "2002-12-22T00:00:00",  # Last
    ]
    ensembles = [
        EnsembleObject(
            name=f"ens_{i}",
            id="id",
            hidden=False,
            experiment_name=f"exp_{i}",
            started_at=t,
        )
        for i, t in enumerate(times)
    ]
    list_widget = EnsembleSelectListWidget(ensembles, 1)
    # Assert experiments are sorted from Last -> First
    assert [list_widget.item(i).text() for i in range(len(ensembles))] == [
        "exp_2 : ens_2",  # Last
        "exp_0 : ens_0",  # Second
        "exp_1 : ens_1",  # First
    ]


def test_that_ensemble_plot_colors_follow_ensemble_selection_color_indexes():
    ensembles = [
        EnsembleObject(
            f"iter-{i}",
            f"id-{i}",
            False,
            "experiment",
            started_at="2026-01-01T00:00:00",
        )
        for i in (3, 2, 1, 0)
    ]
    plot_config = PlotConfig()
    plot_context = PlotContext(
        plot_config,
        ensembles=ensembles,
        ensembles_color_indexes=[0, 1, 2, 3],
        key="FOPR",
        layer=None,
    )
    figure = Figure()

    EnsemblePlot().plot(
        figure,
        plot_context,
        dict.fromkeys(ensembles, pd.DataFrame([[0.1, 0.2], [0.3, 0.4]])),
        pd.DataFrame(),
        {},
        None,
    )

    legend = figure.axes[0].get_legend()
    legend_colors = {
        text.get_text(): handle.get_color()
        for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
    }
    palette = plot_config.line_color_cycle()
    assert legend_colors["experiment : iter-3"] == palette[0][0]
    assert legend_colors["experiment : iter-2"] == palette[1][0]
    assert legend_colors["experiment : iter-1"] == palette[2][0]
    assert legend_colors["experiment : iter-0"] == palette[3][0]
