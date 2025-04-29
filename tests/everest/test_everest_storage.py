from pathlib import Path

import polars as pl
import pytest

from everest.config import EverestConfig
from everest.everest_storage import EverestStorage


@pytest.mark.integration_test
@pytest.mark.parametrize(
    "config_file",
    [
        pytest.param(
            "config_advanced.yml",
            marks=pytest.mark.xdist_group("math_func/config_advanced.yml"),
        ),
        pytest.param(
            "config_minimal.yml",
            marks=pytest.mark.xdist_group("math_func/config_minimal.yml"),
        ),
        pytest.param(
            "config_multiobj.yml",
            marks=pytest.mark.xdist_group("math_func/config_multiobj.yml"),
        ),
    ],
)
def test_csv_export(config_file, cached_example, snapshot):
    config_path, config_file, _, _ = cached_example(f"math_func/{config_file}")
    config = EverestConfig.load_file(Path(config_path) / config_file)

    ever_storage = EverestStorage(output_dir=Path(config.optimization_output_dir))
    ever_storage.init(
        formatted_control_names=config.formatted_control_names,
        objective_functions=config.objective_functions,
        output_constraints=config.output_constraints,
        realizations=config.model.realizations,
    )
    ever_storage.read_from_output_dir()
    combined_df, pert_real_df, batch_df = ever_storage.export_dataframes()

    def _sort_df(df: pl.DataFrame) -> pl.DataFrame:
        df_ = df.select(df.columns)

        sort_rows_by = df_.columns[0 : (min(len(df_.columns), 8))]
        return df_.sort(sort_rows_by)

    snapshot.assert_match(
        _sort_df(combined_df.with_columns(pl.col(pl.Float64).round(4))).write_csv(),
        "combined_df.csv",
    )
    snapshot.assert_match(
        _sort_df(pert_real_df.with_columns(pl.col(pl.Float64).round(4))).write_csv(),
        "pert_real_df.csv",
    )
    snapshot.assert_match(
        _sort_df(batch_df.with_columns(pl.col(pl.Float64).round(4))).write_csv(),
        "batch_df.csv",
    )
