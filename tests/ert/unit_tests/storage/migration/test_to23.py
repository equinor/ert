from enum import StrEnum
from pathlib import Path

import polars as pl

from ert.storage.migration.to23 import migrate

new_localization_keywords = ["east", "north", "radius"]


class ObsType(StrEnum):
    SUMMARY = "summary"
    GEN_OBS = "gen_obs"
    RFT = "rft"


def migrate_old_obs(obs_dict: dict, filename: ObsType) -> pl.DataFrame:
    obs_path = Path("experiments/exp_0/observations")
    obs_path.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame(obs_dict)
    df.write_parquet(obs_path / filename)

    migrate(Path("."))
    return pl.read_parquet(obs_path / filename)


def test_that_old_populated_summary_obs_keep_their_values(use_tmpdir) -> None:
    old_kw_values = {
        "location_x": [None, 1],
        "location_y": [None, 2],
        "location_range": [None, 3],
    }

    result_df = migrate_old_obs(
        {
            "observation_key": ["FOPR_1", "FOPR_2"],
            "response_key": ["summary", "summary"],
            "time": ["2000-01-01", "2000-01-02"],
            "values": [100.0, 200.0],
            "std": [10.0, 20.0],
            "location_x": pl.Series(old_kw_values["location_x"], dtype=pl.Float32),
            "location_y": pl.Series(old_kw_values["location_y"], dtype=pl.Float32),
            "location_range": pl.Series(
                old_kw_values["location_range"], dtype=pl.Float32
            ),
        },
        ObsType.SUMMARY,
    )

    for old_kw, new_kw in zip(old_kw_values, new_localization_keywords, strict=True):
        assert old_kw not in result_df.columns
        assert new_kw in result_df.columns
        assert result_df[new_kw].to_list() == old_kw_values[old_kw]
        assert result_df[new_kw].dtype == pl.Float32


def test_that_localization_columns_are_added_to_unlocalized_summary_obs_dataframes(
    use_tmpdir,
):
    result_df = migrate_old_obs(
        {
            "observation_key": ["FOPR_1", "FOPR_2"],
            "response_key": ["summary", "summary"],
            "time": ["2000-01-01", "2000-01-02"],
            "values": [100.0, 200.0],
            "std": [10.0, 20.0],
        },
        ObsType.SUMMARY,
    )

    for kw in new_localization_keywords:
        assert result_df[kw].to_list() == [None, None]
        assert result_df[kw].dtype == pl.Float32


def old_gen_obs_dict() -> dict:
    return {
        "observation_key": ["FOPR_1", "FOPR_2"],
        "response_key": ["summary", "summary"],
        "report_step": [10, 20],
        "index": [0, 1],
        "observations": [100.0, 200.0],
        "std": [10.0, 20.0],
    }


def test_that_localization_columns_are_added_to_old_gen_obs_dataframes(use_tmpdir):
    result_df = migrate_old_obs(old_gen_obs_dict(), "gen_data")

    for kw in new_localization_keywords:
        assert result_df[kw].to_list() == [None, None]
        assert result_df[kw].dtype == pl.Float32


def test_that_gen_obs_with_localization_columns_remain_unchanged(use_tmpdir):
    old_gen_obs_dict_w_localization_columns = old_gen_obs_dict() | {
        "east": [None, None],
        "north": [None, None],
        "radius": [None, None],
    }
    result_df = migrate_old_obs(
        old_gen_obs_dict_w_localization_columns, ObsType.GEN_OBS
    )
    for key, val in old_gen_obs_dict_w_localization_columns.items():
        assert result_df[key].to_list() == val
    assert len(old_gen_obs_dict_w_localization_columns) == len(result_df.columns)


def old_rft_dict() -> dict:
    return {
        "response_key": ["Foo", "ooF"],
        "observation_key": ["Bar", "raB"],
        "east": [1, 2],
        "north": [3, 4],
        "tvd": [5, 6],
        "observations": [7, 8],
        "std": [9, 10],
    }


def test_that_localization_radius_column_is_added_to_old_rft_dataframes(use_tmpdir):
    result_df = migrate_old_obs(old_rft_dict(), ObsType.RFT)
    assert result_df["radius"].to_list() == [None, None]


def test_that_migration_does_not_change_existing_columns(use_tmpdir):
    result_df = migrate_old_obs(old_rft_dict(), ObsType.RFT)
    for old_kw, val in old_rft_dict().items():
        assert result_df[old_kw].to_list() == val
