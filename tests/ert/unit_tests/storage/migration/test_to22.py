from pathlib import Path

import polars as pl

from ert.storage.migration.to22 import migrate


def migrate_old_summary_obs() -> pl.DataFrame:
    obs_path = Path("experiments/exp_0/observations")
    obs_path.mkdir(parents=True, exist_ok=True)

    summary_df = pl.DataFrame(
        {
            "observation_key": ["FOPR_1", "FOPR_2"],
            "response_key": ["summary", "summary"],
            "time": ["2000-01-01", "2000-01-02"],
            "values": [100.0, 200.0],
            "std": [10.0, 20.0],
        }
    )
    summary_df.write_parquet(obs_path / "summary")

    migrate(Path("."))

    return pl.read_parquet(obs_path / "summary")


def test_that_old_storage_without_location_keywords_gets_them_added(use_tmpdir):
    migrated_df = migrate_old_summary_obs()
    assert migrated_df["location_x"].to_list() == [None, None]
    assert migrated_df["location_y"].to_list() == [None, None]
    assert migrated_df["location_range"].to_list() == [None, None]


def test_that_migrated_location_keywords_have_column_type_f32(use_tmpdir):
    migrated_df = migrate_old_summary_obs()
    assert migrated_df["location_x"].dtype == pl.Float32
    assert migrated_df["location_y"].dtype == pl.Float32
    assert migrated_df["location_range"].dtype == pl.Float32


def test_that_old_storage_with_some_location_keywords_gets_missing_ones_added(
    use_tmpdir,
):
    obs_path = Path("experiments/exp_0/observations")
    obs_path.mkdir(parents=True, exist_ok=True)

    original_location_x = [100.0, None]

    summary_df = pl.DataFrame(
        {
            "observation_key": ["WOPR_1", "WOPR_2"],
            "response_key": ["summary", "summary"],
            "time": ["2000-01-01", "2000-01-02"],
            "values": [150.0, 250.0],
            "std": [15.0, 25.0],
            "location_x": original_location_x,
        }
    )
    summary_df.write_parquet(obs_path / "summary")

    migrate(Path("."))

    migrated_df = pl.read_parquet(obs_path / "summary")

    assert migrated_df["location_x"].to_list() == original_location_x
    assert migrated_df["location_y"].to_list() == [None, None]
    assert migrated_df["location_range"].to_list() == [None, None]


def test_that_new_storage_with_all_location_keywords_remains_unchanged(use_tmpdir):
    obs_path = Path("experiments/exp_0/observations")
    obs_path.mkdir(parents=True, exist_ok=True)

    original_data = {
        "observation_key": ["WBHP_1", "WBHP_2"],
        "response_key": ["summary", "summary"],
        "time": ["2000-01-01", "2000-01-02"],
        "values": [300.0, 400.0],
        "std": [30.0, 40.0],
        "location_x": [100.0, None],
        "location_y": [150.0, None],
        "location_range": [50.0, None],
    }

    summary_df = pl.DataFrame(original_data)
    summary_df.write_parquet(obs_path / "summary")

    migrate(Path("."))

    migrated_df = pl.read_parquet(obs_path / "summary")

    assert len(migrated_df.columns) == 8
    for col in migrated_df.columns:
        assert migrated_df[col].to_list() == original_data[col]
