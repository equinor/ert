from datetime import datetime

import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from ert.gui.tools.plot.plottery.plots.misfits import MisfitsPlot


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
            "key_index": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "response": [12.0, 18.0],
            "error": [2.0, 2.0],
            "observation": [10.0, 20.0],
            "misfit": [1.0, -1.0],
        }
    ).with_columns(pl.col("key_index").cast(pl.Datetime))

    assert_frame_equal(result_df, expected_df)


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
