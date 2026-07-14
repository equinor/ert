from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from ert.gui.plotting.ert_plots.distribution import (
    DistributionPlot,
    _array_is_constant,
)
from ert.gui.plotting.plot_api import EnsembleObject
from ert.gui.plotting.utils.plot_config import PlotConfig
from ert.gui.plotting.utils.plot_context import PlotContext


def _make_ensemble(name: str = "ensemble_1") -> EnsembleObject:
    return EnsembleObject(
        name=name,
        id=name,
        hidden=False,
        experiment_name="experiment_1",
        started_at="2012-12-10T00:00:00",
    )


def _make_context(
    ensembles: list[EnsembleObject],
    *,
    histogram: bool = True,
    gkde_plot: bool = True,
    rug_plot: bool = True,
    by_density: bool = True,
    log_scale: bool = False,
) -> PlotContext:
    context = PlotContext(
        PlotConfig(title="Distribution"),
        ensembles=ensembles,
        ensembles_color_indexes=list(range(len(ensembles))),
        key="A_KEY",
    )
    context.histogram = histogram
    context.gkde_plot = gkde_plot
    context.rug_plot = rug_plot
    context.by_density = by_density
    context.log_scale = log_scale
    return context


def _plot(context: PlotContext, data_map: dict[EnsembleObject, pd.DataFrame]) -> Figure:
    figure = Figure()
    DistributionPlot().plot(figure, context, data_map, pd.DataFrame(), {}, None)
    return figure


def _count_kde_lines(figure: Figure) -> int:
    """A Gaussian KDE curve is a solid line with ~1000 evaluated points."""
    return sum(
        1
        for axes in figure.axes
        for line in axes.get_lines()
        if line.get_marker() in {"", "None"} and np.asarray(line.get_xdata()).size > 100
    )


def _count_histogram_bars(figure: Figure) -> int:
    return sum(
        len(container)
        for axes in figure.axes
        for container in axes.containers
        if isinstance(container, BarContainer)
    )


def _count_rug_marker_lines(figure: Figure) -> int:
    return sum(
        1
        for axes in figure.axes
        for line in axes.get_lines()
        if line.get_marker() == "|"
    )


@pytest.fixture
def single_ensemble() -> EnsembleObject:
    return _make_ensemble()


@pytest.fixture
def varying_data_map(
    single_ensemble: EnsembleObject,
) -> dict[EnsembleObject, pd.DataFrame]:
    return {single_ensemble: pd.DataFrame({0: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})}


def test_that_distribution_plot_shows_message_when_no_plot_option_selected(
    single_ensemble: EnsembleObject,
    varying_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    context = _make_context(
        [single_ensemble], histogram=False, gkde_plot=False, rug_plot=False
    )

    figure = _plot(context, varying_data_map)

    assert any("No plot options selected." in text.get_text() for text in figure.texts)
    assert figure.axes == []


def test_that_only_histogram_is_rendered_when_only_histogram_selected(
    single_ensemble: EnsembleObject,
    varying_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    context = _make_context(
        [single_ensemble], histogram=True, gkde_plot=False, rug_plot=False
    )

    figure = _plot(context, varying_data_map)

    assert _count_histogram_bars(figure) > 0
    assert _count_kde_lines(figure) == 0
    assert _count_rug_marker_lines(figure) == 0


def test_that_only_gkde_is_rendered_when_only_gkde_selected(
    single_ensemble: EnsembleObject,
    varying_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    context = _make_context(
        [single_ensemble], histogram=False, gkde_plot=True, rug_plot=False
    )

    figure = _plot(context, varying_data_map)

    assert _count_kde_lines(figure) == 1
    assert _count_histogram_bars(figure) == 0
    assert _count_rug_marker_lines(figure) == 0


def test_that_only_rug_is_rendered_when_only_rug_selected(
    single_ensemble: EnsembleObject,
    varying_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    context = _make_context(
        [single_ensemble], histogram=False, gkde_plot=False, rug_plot=True
    )

    figure = _plot(context, varying_data_map)

    assert _count_rug_marker_lines(figure) == 1
    assert _count_histogram_bars(figure) == 0
    assert _count_kde_lines(figure) == 0


def test_that_histogram_and_gkde_are_rendered_together_without_rug(
    single_ensemble: EnsembleObject,
    varying_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    context = _make_context(
        [single_ensemble], histogram=True, gkde_plot=True, rug_plot=False
    )

    figure = _plot(context, varying_data_map)

    assert _count_histogram_bars(figure) > 0
    assert _count_kde_lines(figure) == 1
    assert _count_rug_marker_lines(figure) == 0


def test_that_all_three_components_render_when_all_selected(
    single_ensemble: EnsembleObject,
    varying_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    context = _make_context(
        [single_ensemble], histogram=True, gkde_plot=True, rug_plot=True
    )

    figure = _plot(context, varying_data_map)

    assert _count_histogram_bars(figure) > 0
    assert _count_kde_lines(figure) == 1
    assert _count_rug_marker_lines(figure) == 1


def test_that_only_rug_axes_are_created_when_only_rug_selected(
    single_ensemble: EnsembleObject,
    varying_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    context = _make_context(
        [single_ensemble], histogram=False, gkde_plot=False, rug_plot=True
    )

    figure = _plot(context, varying_data_map)

    # One rug axis per ensemble and no separate main plot on top.
    assert len(figure.axes) == 1


def test_that_one_rug_axis_is_created_per_ensemble_for_two_ensembles() -> None:
    ensembles = [_make_ensemble("ensemble_1"), _make_ensemble("ensemble_2")]
    data_map = {
        ensembles[0]: pd.DataFrame({0: [0.1, 0.2, 0.3, 0.4]}),
        ensembles[1]: pd.DataFrame({0: [0.5, 0.6, 0.7, 0.8]}),
    }
    context = _make_context(ensembles, histogram=False, gkde_plot=False, rug_plot=True)

    figure = _plot(context, data_map)

    assert len(figure.axes) == 2
    assert _count_rug_marker_lines(figure) == 2


def test_that_histogram_count_uses_twin_axis_when_gkde_and_not_by_density(
    single_ensemble: EnsembleObject,
    varying_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    context = _make_context(
        [single_ensemble],
        histogram=True,
        gkde_plot=True,
        rug_plot=False,
        by_density=False,
    )

    figure = _plot(context, varying_data_map)

    # A twin y-axis adds a second axes sharing the same subplot position.
    assert len(figure.axes) == 2
    y_labels = {axes.get_ylabel() for axes in figure.axes}
    assert "Count (Histogram)" in y_labels
    assert "Density (Gaussian KDE)" in y_labels


def test_that_single_axis_is_used_when_by_density(
    single_ensemble: EnsembleObject,
    varying_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    context = _make_context(
        [single_ensemble],
        histogram=True,
        gkde_plot=True,
        rug_plot=False,
        by_density=True,
    )

    figure = _plot(context, varying_data_map)

    assert len(figure.axes) == 1
    assert figure.axes[0].get_ylabel() == "Density"


def test_that_histogram_uses_log_x_scale_when_log_scale_enabled(
    single_ensemble: EnsembleObject,
    varying_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    context = _make_context(
        [single_ensemble],
        histogram=True,
        gkde_plot=False,
        rug_plot=False,
        log_scale=True,
    )

    figure = _plot(context, varying_data_map)

    assert figure.axes[0].get_xscale() == "log"


def test_that_gkde_line_is_not_drawn_for_constant_data(
    single_ensemble: EnsembleObject,
) -> None:
    context = _make_context(
        [single_ensemble], histogram=False, gkde_plot=True, rug_plot=False
    )
    data_map = {single_ensemble: pd.DataFrame({0: [1.0, 1.0, 1.0, 1.0]})}

    figure = _plot(context, data_map)

    assert _count_kde_lines(figure) == 0


def test_that_gkde_line_is_not_drawn_for_categorical_data(
    single_ensemble: EnsembleObject,
) -> None:
    context = _make_context(
        [single_ensemble], histogram=False, gkde_plot=True, rug_plot=False
    )
    data_map = {single_ensemble: pd.DataFrame({0: ["cat", "dog", "fish"]})}

    figure = _plot(context, data_map)

    assert _count_kde_lines(figure) == 0


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(pd.DataFrame({0: []}), True, id="empty"),
        pytest.param(pd.DataFrame({0: [3.0, 3.0, 3.0]}), True, id="constant"),
        pytest.param(pd.DataFrame({0: [1.0, 2.0, 3.0]}), False, id="varying"),
    ],
)
def test_that_array_is_constant_detects_empty_constant_and_varying(
    data: pd.DataFrame, expected: bool
) -> None:
    assert bool(_array_is_constant(data[0])) is expected


def test_that_rug_only_plot_uses_log_x_scale_when_log_scale_enabled(
    single_ensemble: EnsembleObject,
    varying_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    context = _make_context(
        [single_ensemble],
        histogram=False,
        gkde_plot=False,
        rug_plot=True,
        log_scale=True,
    )

    figure = _plot(context, varying_data_map)

    assert all(axes.get_xscale() == "log" for axes in figure.axes)


def test_that_all_plots_uses_log_x_scale_when_log_scale_enabled(
    single_ensemble: EnsembleObject,
    varying_data_map: dict[EnsembleObject, pd.DataFrame],
) -> None:
    context = _make_context(
        [single_ensemble],
        histogram=True,
        gkde_plot=True,
        rug_plot=True,
        log_scale=True,
    )

    figure = _plot(context, varying_data_map)

    assert all(axes.get_xscale() == "log" for axes in figure.axes)


def test_that_the_plot_skips_categorical_data_without_raising_error(
    single_ensemble: EnsembleObject,
):

    categorical_df = pd.DataFrame({0: ["cat", "dog", "fish"], 1: [12, 12, 12]})

    context = PlotContext(
        PlotConfig(),
        ensembles=[single_ensemble],
        ensembles_color_indexes=[0],
        key="animal_type",
        layer=None,
    )
    figure = _plot(context, {single_ensemble: categorical_df})

    assert _count_kde_lines(figure) == 0
    assert _count_histogram_bars(figure) == 0
    assert _count_rug_marker_lines(figure) == 0
