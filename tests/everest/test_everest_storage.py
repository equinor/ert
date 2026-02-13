from pathlib import Path

import polars as pl
import pytest

from ert.storage import open_storage
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

    experiment = EverestStorage.get_everest_experiment(
        storage_path=config.storage_dir,
    )
    combined_df, pert_real_df, batch_df = experiment.export_dataframes()

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


@pytest.mark.integration_test
@pytest.mark.parametrize(
    ("config_file", "responses", "objectives", "constraints", "gen_data_only"),
    [
        pytest.param(
            "config_advanced.yml",
            {"gen_data", "everest_constraints", "everest_objectives"},
            {"distance"},
            {"x-0_coord"},
            {"distance_nonobj"},
            marks=pytest.mark.xdist_group("math_func/config_advanced.yml"),
        ),
        pytest.param(
            "config_minimal.yml",
            {"everest_objectives"},
            {"distance"},
            set(),
            set(),
            marks=pytest.mark.xdist_group("math_func/config_minimal.yml"),
        ),
        pytest.param(
            "config_multiobj.yml",
            {"everest_objectives"},
            {"distance_p", "distance_q"},
            set(),
            set(),
            marks=pytest.mark.xdist_group("math_func/config_multiobj.yml"),
        ),
    ],
)
def test_everest_data_stored_in_ert_local_storage(
    config_file, responses, objectives, constraints, gen_data_only, cached_example
):
    config_path, config_file, _, _ = cached_example(f"math_func/{config_file}")

    config = EverestConfig.load_file(Path(config_path) / config_file)
    with open_storage(config.storage_dir, mode="r") as storage:
        experiment = next(s for s in storage._experiments.values())
        assert set(experiment.response_info.keys()) == responses

        response_type_mapping = experiment.response_type_to_response_keys
        assert set(response_type_mapping.get("gen_data", [])) == gen_data_only
        assert set(response_type_mapping.get("everest_constraints", [])) == constraints
        assert set(response_type_mapping.get("everest_objectives", [])) == objectives

        local_storage_params = [
            cfg.input_key for cfg in experiment.parameter_configuration.values()
        ]

        formatted_control_names = [
            name
            for control_config in config.controls
            for name in control_config.formatted_control_names
        ]

        assert local_storage_params == formatted_control_names
