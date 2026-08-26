from datetime import datetime
from itertools import starmap

import pandas as pd
import polars as pl
import pytest
from matplotlib.figure import Figure
from polars.testing import assert_frame_equal

from ert.gui.plotting.ert_plots.misfits import MisfitsPlot
from ert.gui.plotting.plot_api import EnsembleObject, PlotApiKeyDefinition
from ert.gui.plotting.utils import PlotConfig, PlotContext


@pytest.fixture
def make_ensemble():
    def _make(name: str, id_: str | None = None) -> EnsembleObject:
        return EnsembleObject(
            name,
            id_ if id_ is not None else name,
            False,
            "experiment",
            "2026-01-01T00:00:00",
        )

    return _make


@pytest.fixture
def make_plot_context():
    def _make(
        ensembles: list[EnsembleObject],
        *,
        key: str = "FOPR",
        plot_config: PlotConfig | None = None,
    ) -> PlotContext:
        return PlotContext(
            plot_config if plot_config is not None else PlotConfig(),
            ensembles=ensembles,
            ensembles_color_indexes=list(range(len(ensembles))),
            key=key,
            layer=None,
        )

    return _make


@pytest.fixture
def make_key_def():
    def _make(
        *, key: str = "FOPR", data_origin: str = "summary"
    ) -> PlotApiKeyDefinition:
        return PlotApiKeyDefinition(
            key,
            index_type=None,
            metadata={"data_origin": data_origin},
            observations=True,
        )

    return _make


def test_that_misfits_plot_is_empty_when_observations_are_missing(
    make_ensemble, make_plot_context, make_key_def
):
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble])
    key_def = make_key_def()
    figure = Figure()

    MisfitsPlot().plot(
        figure,
        plot_context,
        {
            ensemble: pd.DataFrame(
                {"2023-01-01": [12.0]},
                index=pd.Index([0], name="Realization"),
            )
        },
        observation_data=pd.DataFrame(),
        std_dev_images={},
        obs_loc=None,
        key_def=key_def,
    )

    assert len(figure.axes) == 1
    assert figure.axes[0].texts[0].get_text() == "No observations available"


def test_that_misfit_conversion_for_summary_casts_key_index_to_datetime():
    ensemble_df = pd.DataFrame(
        {"2023-01-01": [10.0], "2023-01-02": [20.0]},
        index=pd.Index([0], name="Realization"),
    )
    ensemble_to_data_map = {("ens1", "id1"): ensemble_df}

    observation_data = pd.DataFrame(
        data={
            0: [1.0, 10.0, "2023-01-01"],
            1: [2.0, 20.0, "2023-01-02"],
        },
        index=["STD", "OBS", "key_index"],
    )

    result = MisfitsPlot._wide_pandas_to_long_polars_with_misfits(
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
        response_type="summary",
    )

    result_df = result["ens1", "id1"]
    assert result_df["key_index"].dtype == pl.Datetime


def test_that_misfit_conversion_for_summary_converts_to_equivalent_long_polars_df():
    ensemble_df = pd.DataFrame(
        {"2023-01-01": [12.0], "2023-01-02": [18.0]},
        index=pd.Index([0], name="Realization"),
    )
    ensemble_to_data_map = {("ens1", "id1"): ensemble_df}

    observation_data = pd.DataFrame(
        data={
            0: [2.0, 10.0, "2023-01-01"],
            1: [2.0, 20.0, "2023-01-02"],
        },
        index=["STD", "OBS", "key_index"],
    )

    result = MisfitsPlot._wide_pandas_to_long_polars_with_misfits(
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
        response_type="summary",
    )

    result_df = result["ens1", "id1"]

    expected_df = pl.DataFrame(
        {
            "Realization": [0, 0],
            "key_index": [datetime(2023, 1, 1), datetime(2023, 1, 2)],  # ruff: ignore[call-datetime-without-tzinfo]
            "response": [12.0, 18.0],
            "error": [2.0, 2.0],
            "observation": [10.0, 20.0],
            "misfit": [1.0, -1.0],
        }
    ).with_columns(pl.col("key_index").cast(pl.Datetime))

    assert_frame_equal(result_df, expected_df)


@pytest.mark.parametrize(
    ("data_origin", "key", "ensemble_specs", "ensemble_frames", "observation_data"),
    [
        pytest.param(
            "summary",
            "FOPR",
            [("ensemble",)],
            [{"2023-01-01": [12.0]}],
            {0: [1.0, 10.0, "2023-01-02"]},
            id="summary_key_index_mismatch",
        ),
        pytest.param(
            "seismic",
            "SEISMIC_KEY",
            [
                ("ensemble_empty_1", "ensemble_empty_id_1"),
                ("ensemble_empty_2", "ensemble_empty_id_2"),
            ],
            [{"99": [1.0], "100": [2.0]}, {"99": [3.0], "100": [4.0]}],
            {0: [10.0, 100.0, "0"], 1: [10.0, 200.0, "5"]},
            id="seismic_key_index_mismatch_all_ensembles",
        ),
    ],
)
def test_that_misfits_plot_is_empty_when_no_misfit_data_is_available(
    data_origin,
    key,
    ensemble_specs,
    ensemble_frames,
    observation_data,
    make_ensemble,
    make_plot_context,
    make_key_def,
):
    ensembles = list(starmap(make_ensemble, ensemble_specs))
    plot_context = make_plot_context(ensembles, key=key)
    key_def = make_key_def(key=key, data_origin=data_origin)
    figure = Figure()

    MisfitsPlot().plot(
        figure,
        plot_context,
        {
            ensemble: pd.DataFrame(frame, index=pd.Index([0], name="Realization"))
            for ensemble, frame in zip(ensembles, ensemble_frames, strict=True)
        },
        observation_data=pd.DataFrame(
            data=observation_data,
            index=["STD", "OBS", "key_index"],
        ),
        std_dev_images={},
        obs_loc=None,
        key_def=key_def,
    )

    assert len(figure.axes) == 1
    assert figure.axes[0].texts[0].get_text() == "No misfit data available"


def test_that_misfit_conversion_for_gen_data_casts_key_index_to_uint16():
    ensemble_df = pd.DataFrame(
        {"10": [100.0], "20": [200.0]}, index=pd.Index([0], name="Realization")
    )
    ensemble_to_data_map = {("ens1", "id1"): ensemble_df}

    observation_data = pd.DataFrame(
        data={
            0: [10.0, 100.0, "10"],
            1: [20.0, 200.0, "20"],
        },
        index=["STD", "OBS", "key_index"],
    )

    result = MisfitsPlot._wide_pandas_to_long_polars_with_misfits(
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
        response_type="gen_data",
    )

    result_df = result["ens1", "id1"]
    assert result_df["key_index"].dtype == pl.UInt16


def test_that_misfit_conversion_for_gen_data_converts_to_equivalent_long_polars_df():
    ensemble_df = pd.DataFrame(
        {"10": [110.0], "20": [180.0]}, index=pd.Index([0], name="Realization")
    )
    ensemble_to_data_map = {("ens1", "id1"): ensemble_df}

    observation_data = pd.DataFrame(
        data={
            0: [10.0, 100.0, "10"],
            1: [10.0, 200.0, "20"],
        },
        index=["STD", "OBS", "key_index"],
    )

    result = MisfitsPlot._wide_pandas_to_long_polars_with_misfits(
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
        response_type="gen_data",
    )

    result_df = result["ens1", "id1"]

    expected_df = pl.DataFrame(
        {
            "Realization": [0, 0],
            "key_index": [10, 20],
            "response": [110.0, 180.0],
            "error": [10.0, 10.0],
            "observation": [100.0, 200.0],
            "misfit": [1.0, -4.0],
        }
    ).with_columns(pl.col("key_index").cast(pl.UInt16))

    assert_frame_equal(result_df, expected_df)


def test_that_box_and_scatter_plot_is_being_plotted_for_summary_data(
    make_ensemble, make_plot_context, make_key_def
):
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble])
    plot_context.scatter_plot = True  # box is true by default

    key_def = make_key_def()
    figure = Figure()

    MisfitsPlot().plot(
        figure,
        plot_context,
        {
            ensemble: pd.DataFrame(
                {"2023-01-01": [12.0], "2023-01-02": [18.0]},
                index=pd.Index([0], name="Realization"),
            )
        },
        observation_data=pd.DataFrame(
            data={
                0: [2.0, 10.0, "2023-01-01"],
                1: [2.0, 20.0, "2023-01-02"],
            },
            index=["STD", "OBS", "key_index"],
        ),
        std_dev_images={},
        obs_loc=None,
        key_def=key_def,
    )

    assert len(figure.axes) == 1
    axes = figure.axes[0]
    assert len(axes.collections) == 1  # scatter
    assert len(axes.patches) == 3  # 2 boxes + background patch


def test_that_mean_gets_plotted_for_summary_data_when_enabled(
    make_ensemble, make_plot_context, make_key_def
):
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble])
    plot_context.box_plot = False
    plot_context.mean = True

    key_def = make_key_def()
    figure = Figure()

    MisfitsPlot().plot(
        figure,
        plot_context,
        {
            ensemble: pd.DataFrame(
                {"2023-01-01": [12.0], "2023-01-02": [18.0]},
                index=pd.Index([0], name="Realization"),
            )
        },
        observation_data=pd.DataFrame(
            data={
                0: [2.0, 10.0, "2023-01-01"],
                1: [2.0, 20.0, "2023-01-02"],
            },
            index=["STD", "OBS", "key_index"],
        ),
        std_dev_images={},
        obs_loc=None,
        key_def=key_def,
    )

    assert len(figure.axes) == 1
    axes = figure.axes[0]
    axes_lines = axes.get_lines()
    assert len(axes_lines) == 2  # mean line + hline


@pytest.mark.parametrize("enabled", [True, False])
def test_that_legend_items_for_summary_data_is_toggleable(
    enabled, make_ensemble, make_plot_context, make_key_def
):
    legend_items = [
        "Mean",
        "Median",
        "Outliers",
        "Scatter points",
    ]
    ensemble = make_ensemble("ensemble")
    plot_context = make_plot_context([ensemble])
    plot_context.mean = enabled
    plot_context.scatter_plot = enabled
    plot_context.box_plot = enabled

    key_def = make_key_def()
    figure = Figure()

    MisfitsPlot().plot(
        figure,
        plot_context,
        {
            ensemble: pd.DataFrame(
                {"2023-01-01": [12.0], "2023-01-02": [18.0]},
                index=pd.Index([0], name="Realization"),
            )
        },
        observation_data=pd.DataFrame(
            data={
                0: [2.0, 10.0, "2023-01-01"],
                1: [2.0, 20.0, "2023-01-02"],
            },
            index=["STD", "OBS", "key_index"],
        ),
        std_dev_images={},
        obs_loc=None,
        key_def=key_def,
    )

    axes = figure.axes[0]
    legend_texts = [text.get_text() for text in axes.get_legend().get_texts()]
    for item in legend_items:
        assert (item in legend_texts) == enabled


def test_that_misfit_conversion_for_seismic_casts_key_index_to_int32():
    ensemble_df = pd.DataFrame(
        {"0": [110.0], "5": [180.0]}, index=pd.Index([0], name="Realization")
    )
    ensemble_to_data_map = {("ens1", "id1"): ensemble_df}

    observation_data = pd.DataFrame(
        data={
            0: [10.0, 100.0, "0"],
            1: [10.0, 200.0, "5"],
        },
        index=["STD", "OBS", "key_index"],
    )

    result = MisfitsPlot._wide_pandas_to_long_polars_with_misfits(
        ensemble_to_data_map=ensemble_to_data_map,
        observation_data=observation_data,
        response_type="seismic",
    )

    result_df = result["ens1", "id1"]
    assert result_df["key_index"].dtype == pl.Int32


def test_that_seismic_misfit_plot_draws_one_box_per_ensemble_with_ensemble_name(
    make_ensemble, make_plot_context, make_key_def
):
    ensembles = [
        make_ensemble("ensemble_a", "ensemble_id_a"),
        make_ensemble("ensemble_b", "ensemble_id_b"),
    ]
    plot_context = make_plot_context(
        ensembles, key="SEISMIC_KEY", plot_config=PlotConfig(title=None)
    )
    key_def = make_key_def(key="SEISMIC_KEY", data_origin="seismic")
    figure = Figure()

    MisfitsPlot().plot(
        figure,
        plot_context,
        {
            ensembles[0]: pd.DataFrame(
                {"0": [110.0], "5": [180.0]},
                index=pd.Index([0], name="Realization"),
            ),
            ensembles[1]: pd.DataFrame(
                {"0": [90.0], "5": [220.0]},
                index=pd.Index([0], name="Realization"),
            ),
        },
        observation_data=pd.DataFrame(
            data={
                0: [10.0, 100.0, "0"],
                1: [10.0, 200.0, "5"],
            },
            index=["STD", "OBS", "key_index"],
        ),
        std_dev_images={},
        obs_loc=None,
        key_def=key_def,
    )

    assert len(figure.axes) == 1
    axes = figure.axes[0]
    xtick_labels = [t.get_text() for t in axes.get_xticklabels()]
    assert xtick_labels == ["ensemble_a", "ensemble_b"]
    # One box patch per ensemble (no per-timestep background shading here).
    assert len(axes.patches) == 2
    assert axes.get_xlabel() == "Ensemble name"
    assert axes.get_ylabel() == "Mean signed Chi-squared misfit"
    assert "Mean signed Chi-squared misfit per ensemble" in axes.get_title()


def test_that_seismic_misfit_plot_annotates_median_to_each_box_when_box_plot_enabled(
    make_ensemble, make_plot_context, make_key_def
):
    ensembles = [
        make_ensemble("ensemble_a", "ensemble_id_a"),
        make_ensemble("ensemble_b", "ensemble_id_b"),
    ]
    plot_context = make_plot_context(ensembles, key="SEISMIC_KEY")
    plot_context.box_plot = True

    key_def = make_key_def(key="SEISMIC_KEY", data_origin="seismic")
    figure = Figure()

    MisfitsPlot().plot(
        figure,
        plot_context,
        {
            ensembles[0]: pd.DataFrame(
                {"0": [110.0], "5": [180.0]},
                index=pd.Index([0], name="Realization"),
            ),
            ensembles[1]: pd.DataFrame(
                {"0": [90.0], "5": [220.0]},
                index=pd.Index([0], name="Realization"),
            ),
        },
        observation_data=pd.DataFrame(
            data={
                0: [10.0, 100.0, "0"],
                1: [10.0, 200.0, "5"],
            },
            index=["STD", "OBS", "key_index"],
        ),
        std_dev_images={},
        obs_loc=None,
        key_def=key_def,
    )

    axes = figure.axes[0]
    annotation_texts = sorted(t.get_text() for t in axes.texts)
    assert annotation_texts == ["-1.5", "1.5"]


def test_that_seismic_misfit_plot_omits_median_annotations_when_box_plot_disabled(
    make_ensemble, make_plot_context, make_key_def
):
    ensemble = make_ensemble("ensemble", "ensemble_id")
    plot_context = make_plot_context([ensemble], key="SEISMIC_KEY")
    plot_context.box_plot = False
    plot_context.scatter_plot = True

    key_def = make_key_def(key="SEISMIC_KEY", data_origin="seismic")
    figure = Figure()

    MisfitsPlot().plot(
        figure,
        plot_context,
        {
            ensemble: pd.DataFrame(
                {"0": [110.0], "5": [180.0]},
                index=pd.Index([0], name="Realization"),
            )
        },
        observation_data=pd.DataFrame(
            data={
                0: [10.0, 100.0, "0"],
                1: [10.0, 200.0, "5"],
            },
            index=["STD", "OBS", "key_index"],
        ),
        std_dev_images={},
        obs_loc=None,
        key_def=key_def,
    )

    axes = figure.axes[0]
    assert len(axes.texts) == 0


def test_that_seismic_misfit_plot_skips_ensembles_with_no_misfit_data(
    make_ensemble, make_plot_context, make_key_def
):
    ensembles = [
        make_ensemble("ensemble", "ensemble_id"),
        make_ensemble("ensemble_empty", "ensemble_empty_id"),
    ]
    plot_context = make_plot_context(
        ensembles, key="SEISMIC_KEY", plot_config=PlotConfig(title=None)
    )
    key_def = make_key_def(key="SEISMIC_KEY", data_origin="seismic")

    figure = Figure()

    MisfitsPlot().plot(
        figure,
        plot_context,
        {
            ensembles[0]: pd.DataFrame(
                {"0": [110.0], "5": [180.0]},
                index=pd.Index([0], name="Realization"),
            ),
            ensembles[1]: pd.DataFrame(
                {"99": [1.0], "100": [2.0]},
                index=pd.Index([0], name="Realization"),
            ),
        },
        observation_data=pd.DataFrame(
            data={
                0: [10.0, 100.0, "0"],
                1: [10.0, 200.0, "5"],
            },
            index=["STD", "OBS", "key_index"],
        ),
        std_dev_images={},
        obs_loc=None,
        key_def=key_def,
    )

    assert len(figure.axes) == 1
    assert all(
        text.get_text() != "No misfit data available" for text in figure.axes[0].texts
    )
    xtick_labels = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]
    assert xtick_labels == ["ensemble"]
    assert len(figure.axes[0].patches) == 1
    assert figure.axes[0].get_xlim() == (-0.5, 0.5)
