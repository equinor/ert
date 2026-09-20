import datetime
import json
import re
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ert.storage.migration.to28 import (
    location_metadata_schema,
    migrate,
    observation_schema,
    original_response_schema,
    response_schema,
    transform,
)


def _to_original_response(
    responses: pl.DataFrame, observations: pl.DataFrame, location_metadata: pl.DataFrame
) -> pl.DataFrame:
    observations_with_metadata = (
        observations.join(
            location_metadata,
            on=["east", "north", "tvd"],
            how="left",
        )
        .select(
            [
                "east",
                "north",
                "tvd",
                pl.col("zone").alias("expected_zone"),
                "actual_zones",
                "well_connection_cell",
            ]
        )
        .unique(maintain_order=True)
    )

    result = responses.join(
        observations_with_metadata,
        on="well_connection_cell",
        how="left",
    )

    is_zone_valid = pl.col("expected_zone").is_null() | pl.col("expected_zone").is_in(
        pl.col("actual_zones")
    )
    result = result.filter(is_zone_valid)

    return result.select(
        [
            "realization",
            "response_key",
            "well",
            "date",
            "property",
            "time",
            "depth",
            "values",
            pl.col("expected_zone").alias("zone"),
            pl.col("east").cast(pl.Float32),
            pl.col("north").cast(pl.Float32),
            pl.col("tvd").cast(pl.Float32),
            pl.col("well_connection_cell").arr.get(0).alias("i"),
            pl.col("well_connection_cell").arr.get(1).alias("j"),
            pl.col("well_connection_cell").arr.get(2).alias("k"),
        ]
    )


def assert_schema(df: pl.DataFrame, schema: dict[str, Any]) -> pl.DataFrame:
    if df.schema != schema:
        msg = f"Expected schema {schema}, got {df.schema}."
        raise AssertionError(msg)
    return df


def assert_roundtrip(
    observations_df: pl.DataFrame, original_rft_response: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rft_response_df, location_metadata_df = transform(
        observations_df, original_rft_response
    )

    assert_schema(rft_response_df, response_schema())
    assert_schema(location_metadata_df, location_metadata_schema())

    combined_df = _to_original_response(
        rft_response_df, observations_df, location_metadata_df
    )

    assert_schema(combined_df, original_response_schema())
    assert_frame_equal(combined_df, original_rft_response)

    return rft_response_df, location_metadata_df


base_date = datetime.date(2000, 1, 1)
base_observation_truncated = {
    "observation_key": "OBS1",
    "response_key": "WELL:2000-01-01:PRESSURE",
    "observations": 1500.0,
    "east": 501.0,
    "north": 1001.0,
    "tvd": 901.0,
    "zone": "zone1",
}
base_original_response_data = {
    "realization": 0,
    "response_key": "WELL:2000-01-01:PRESSURE",
    "well": "WELL",
    "date": base_date.isoformat(),
    "property": "PRESSURE",
    "time": base_date,
    "depth": 1000.0,
    "values": 200.0,
    "zone": "zone1",
    "east": 501.0,
    "north": 1001.0,
    "tvd": 901.0,
    "i": 10,
    "j": 11,
    "k": 12,
}

expected_base_response_data = {
    "realization": 0,
    "response_key": "WELL:2000-01-01:PRESSURE",
    "well": "WELL",
    "date": base_date.isoformat(),
    "property": "PRESSURE",
    "time": base_date,
    "depth": 1000.0,
    "values": 200.0,
    "well_connection_cell": [10, 11, 12],
}

expected_base_location_metadata_data = {
    "east": 501.0,
    "north": 1001.0,
    "tvd": 901.0,
    "actual_zones": ["zone1"],
    "well_connection_cell": [10, 11, 12],
}


def setup_storage(tmp_path, original_response_df, observations_df):
    experiment_hash = "experiment_hash"
    ensemble_hash = "ensemble_hash"

    root_path = tmp_path / "project"

    ensemble_path = root_path / "ensembles" / ensemble_hash
    ensemble_path.mkdir(parents=True)

    realization_path = ensemble_path / "realization-0"
    realization_path.mkdir(parents=True)

    index_path = ensemble_path / "index.json"
    index_path.write_text(
        json.dumps({"experiment_id": experiment_hash}),
        encoding="utf-8",
    )

    obs_path = root_path / "experiments" / experiment_hash / "observations"
    obs_path.mkdir(parents=True)

    rft_obs_path = obs_path / "rft"
    response_path = realization_path / "rft.parquet"
    location_metadata_path = (
        realization_path / "rft_observation_location_metadata.parquet"
    )

    original_response_df.write_parquet(response_path)
    if observations_df is not None:
        observations_df.write_parquet(rft_obs_path)

    return root_path, response_path, location_metadata_path


def test_that_migrating_storage_to_28_uses_empty_dataframes_on_invalid_response_schema(
    tmp_path, caplog
):
    obs_df = pl.DataFrame([{**base_observation_truncated}], schema=observation_schema())
    original_response_df = pl.DataFrame(
        {
            "response_key": [],
            "time": [],
            "depth": [],
            "values": [],
            "east": [],
            "north": [],
            "tvd": [],
        }
    )

    root, response_path, location_metadata_path = setup_storage(
        tmp_path, original_response_df, obs_df
    )

    migrate(root)
    warning = re.escape(
        f"Schema error on migrating {response_path}: Unexpected schema for rft.parquet"
    )
    assert re.search(warning, caplog.text) is not None

    response_df = pl.read_parquet(response_path)
    location_metadata_df = pl.read_parquet(location_metadata_path)

    expected_response_df = pl.DataFrame(schema=response_schema())
    expected_location_metadata_df = pl.DataFrame(schema=location_metadata_schema())

    assert_schema(response_df, response_schema())
    assert_schema(location_metadata_df, location_metadata_schema())
    assert_frame_equal(response_df, expected_response_df)
    assert_frame_equal(location_metadata_df, expected_location_metadata_df)


def test_that_migrating_storage_to_28_migrates_on_missing_observations(
    tmp_path, caplog
):
    obs_df = None
    original_response_df = pl.DataFrame(
        [{**base_original_response_data}], schema=original_response_schema()
    )

    root, response_path, location_metadata_path = setup_storage(
        tmp_path, original_response_df, obs_df
    )

    migrate(root)

    warning = "Assuming empty observations"
    assert warning in caplog.text

    response_df = pl.read_parquet(response_path)
    location_metadata_df = pl.read_parquet(location_metadata_path)

    expected_response_df = pl.DataFrame(
        [
            {
                **expected_base_response_data,
            }
        ],
        schema=response_schema(),
    )
    expected_location_metadata_df = pl.DataFrame(schema=location_metadata_schema())

    assert_schema(response_df, response_schema())
    assert_schema(location_metadata_df, location_metadata_schema())
    assert_frame_equal(response_df, expected_response_df)
    assert_frame_equal(location_metadata_df, expected_location_metadata_df)


def test_that_migrating_storage_to_28_uses_empty_dataframes_on_invalid_obs_schema(
    tmp_path, caplog
):
    obs_df = pl.DataFrame(
        {
            "unexpected_column": "cat",
            "north": 13.0,
        }
    )

    original_response_df = pl.DataFrame(
        [{**base_original_response_data}], schema=original_response_schema()
    )

    root, response_path, location_metadata_path = setup_storage(
        tmp_path, original_response_df, obs_df
    )

    migrate(root)

    warning = re.escape(
        f"Schema error on migrating {response_path}: Unexpected schema for observations"
    )
    assert re.search(warning, caplog.text) is not None

    response_df = pl.read_parquet(response_path)
    location_metadata_df = pl.read_parquet(location_metadata_path)

    expected_response_df = pl.DataFrame(schema=response_schema())
    expected_location_metadata_df = pl.DataFrame(schema=location_metadata_schema())

    assert_schema(response_df, response_schema())
    assert_schema(location_metadata_df, location_metadata_schema())
    assert_frame_equal(response_df, expected_response_df)
    assert_frame_equal(location_metadata_df, expected_location_metadata_df)


def test_that_migrating_storage_to_28_does_not_fail_on_empty_dataframes(tmp_path):
    obs_df = pl.DataFrame(schema=observation_schema())

    original_response_df = pl.DataFrame(schema=original_response_schema())

    root, response_path, location_metadata_path = setup_storage(
        tmp_path, original_response_df, obs_df
    )
    migrate(root)

    response_df = pl.read_parquet(response_path)
    location_metadata_df = pl.read_parquet(location_metadata_path)

    expected_response_df = pl.DataFrame(schema=response_schema())
    expected_location_metadata_df = pl.DataFrame(schema=location_metadata_schema())

    assert_schema(response_df, response_schema())
    assert_schema(location_metadata_df, location_metadata_schema())
    assert_frame_equal(response_df, expected_response_df)
    assert_frame_equal(location_metadata_df, expected_location_metadata_df)


def test_that_migrating_storage_to_28_stores_responses_and_location_metadata(tmp_path):
    obs_df = pl.DataFrame([{**base_observation_truncated}], schema=observation_schema())
    original_response_df = pl.DataFrame(
        [{**base_original_response_data}], schema=original_response_schema()
    )

    root, response_path, location_metadata_path = setup_storage(
        tmp_path, original_response_df, obs_df
    )
    migrate(root)

    response_df = pl.read_parquet(response_path)
    location_metadata_df = pl.read_parquet(location_metadata_path)

    expected_response_df = pl.DataFrame(
        [
            {
                **expected_base_response_data,
            }
        ],
        schema=response_schema(),
    )

    expected_location_metadata_df = pl.DataFrame(
        [
            {
                **expected_base_location_metadata_data,
            }
        ],
        schema=location_metadata_schema(),
    )

    assert_schema(response_df, response_schema())
    assert_schema(location_metadata_df, location_metadata_schema())
    assert_frame_equal(response_df, expected_response_df)
    assert_frame_equal(location_metadata_df, expected_location_metadata_df)


def test_that_migrating_storage_to_28_keeps_roundtrip_intact():
    observations_df = pl.DataFrame(
        [{**base_observation_truncated}], schema=observation_schema()
    )
    original_rft_response = pl.DataFrame(
        [{**base_original_response_data}], schema=original_response_schema()
    )
    assert_roundtrip(observations_df, original_rft_response)


def test_that_migrating_storage_to_28_keeps_responses_independent_from_observations():
    observation_property = "SURPRISE"
    observation_response_key = f"WELL:2000-01-01:{observation_property}"
    observation_tvd = 911.0
    observations_df = pl.DataFrame(
        [
            {
                **base_observation_truncated,
                "response_key": observation_response_key,
                "tvd": observation_tvd,
            }
        ],
        schema=observation_schema(),
    )
    # this response shouldn't have happened from provided observations. Making those
    # inconsistent for the test. So no roundtrip, as it will fail.
    original_rft_response = pl.DataFrame(
        [{**base_original_response_data}], schema=original_response_schema()
    )
    expected_response_df = pl.DataFrame(
        [
            {
                **expected_base_response_data,
            }
        ],
        schema=response_schema(),
    )

    expected_location_metadata_df = pl.DataFrame(
        [
            {
                **expected_base_location_metadata_data,
                "tvd": observation_tvd,
                "actual_zones": [],
                "well_connection_cell": None,
            }
        ],
        schema=location_metadata_schema(),
    )

    response_ds, location_metadata_ds = transform(
        observations_df, original_rft_response
    )
    assert_frame_equal(response_ds, expected_response_df)
    assert_frame_equal(location_metadata_ds, expected_location_metadata_df)


def test_that_migrating_storage_to_28_merges_response_rows():
    tvd1 = 901.0
    tvd2 = 902.0
    observations_df = pl.DataFrame(
        [
            {**base_observation_truncated, "observation_key": "OBS1", "tvd": tvd1},
            {**base_observation_truncated, "observation_key": "OBS2", "tvd": tvd2},
        ],
        schema=observation_schema(),
    )
    original_rft_response = pl.DataFrame(
        [
            {**base_original_response_data, "tvd": tvd1},
            {**base_original_response_data, "tvd": tvd2},
        ],
        schema=original_response_schema(),
    )

    expected_response_df = pl.DataFrame(
        [
            {
                **expected_base_response_data,
            }
        ],
        schema=response_schema(),
    )

    expected_location_metadata_df = pl.DataFrame(
        [
            {
                **expected_base_location_metadata_data,
                "tvd": tvd1,
            },
            {
                **expected_base_location_metadata_data,
                "tvd": tvd2,
            },
        ],
        schema=location_metadata_schema(),
    )

    response_ds, location_metadata_ds = assert_roundtrip(
        observations_df, original_rft_response
    )
    assert_frame_equal(response_ds, expected_response_df)
    assert_frame_equal(location_metadata_ds, expected_location_metadata_df)


def test_that_migrating_storage_to_28_merges_metadata_rows():
    property1 = "PRESSURE"
    property2 = "TEMPERATURE"
    response_key1 = f"WELL:2000-01-01:{property1}"
    response_key2 = f"WELL:2000-01-01:{property2}"
    observation_df = pl.DataFrame(
        [
            {
                **base_observation_truncated,
                "observation_key": "OBS1",
                "response_key": response_key1,
            },
            {
                **base_observation_truncated,
                "observation_key": "OBS2",
                "response_key": response_key2,
            },
        ],
        schema=observation_schema(),
    )
    original_rft_response = pl.DataFrame(
        [
            {
                **base_original_response_data,
                "response_key": response_key1,
                "property": property1,
            },
            {
                **base_original_response_data,
                "response_key": response_key2,
                "property": property2,
            },
        ],
        schema=original_response_schema(),
    )
    expected_response_df = pl.DataFrame(
        [
            {
                **expected_base_response_data,
                "response_key": response_key1,
                "property": property1,
            },
            {
                **expected_base_response_data,
                "response_key": response_key2,
                "property": property2,
            },
        ],
        schema=response_schema(),
    )

    expected_location_metadata_df = pl.DataFrame(
        [
            {
                **expected_base_location_metadata_data,
            }
        ],
        schema=location_metadata_schema(),
    )

    response_ds, location_metadata_ds = assert_roundtrip(
        observation_df, original_rft_response
    )
    assert_frame_equal(response_ds, expected_response_df)
    assert_frame_equal(location_metadata_ds, expected_location_metadata_df)


def test_that_migrating_storage_to_28_gives_well_connection_only_to_fitting_location():
    depth1 = 1000.0
    depth2 = 1025.0
    observations_df = pl.DataFrame(
        [
            {**base_observation_truncated},
        ],
        schema=observation_schema(),
    )
    original_rft_response = pl.DataFrame(
        [
            {**base_original_response_data, "depth": depth1, "k": 12},
            {
                **base_original_response_data,
                "depth": depth2,
                "k": 13,
                "zone": None,
                "east": None,
                "north": None,
                "tvd": None,
            },
        ],
        schema=original_response_schema(),
    )
    expected_response_df = pl.DataFrame(
        [
            {
                **expected_base_response_data,
                "depth": depth1,
                "well_connection_cell": [10, 11, 12],
            },
            {
                **expected_base_response_data,
                "depth": depth2,
                "well_connection_cell": [10, 11, 13],
            },
        ],
        schema=response_schema(),
    )

    expected_location_metadata_df = pl.DataFrame(
        [
            {
                **expected_base_location_metadata_data,
                "well_connection_cell": [10, 11, 12],
            }
        ],
        schema=location_metadata_schema(),
    )

    response_ds, location_metadata_ds = assert_roundtrip(
        observations_df, original_rft_response
    )
    assert_frame_equal(response_ds, expected_response_df)
    assert_frame_equal(location_metadata_ds, expected_location_metadata_df)


def test_that_migrating_storage_to_28_restores_location_disabled_due_to_zone_mismatch():
    observations_df = pl.DataFrame(
        [
            {
                **base_observation_truncated,
                "zone": "zone_not_fitting_into_zonemap",
            },
        ],
        schema=observation_schema(),
    )
    original_rft_response = pl.DataFrame([], schema=original_response_schema())

    expected_response_df = pl.DataFrame([], schema=response_schema())
    expected_location_metadata_df = pl.DataFrame(
        [
            {
                **expected_base_location_metadata_data,
                "actual_zones": [],
                "well_connection_cell": None,
            }
        ],
        schema=location_metadata_schema(),
    )

    response_ds, location_metadata_ds = assert_roundtrip(
        observations_df, original_rft_response
    )
    assert_frame_equal(response_ds, expected_response_df)
    assert_frame_equal(location_metadata_ds, expected_location_metadata_df)


def test_that_migrating_storage_to_28_uses_observations_for_data_on_missing_zone():
    observations_df = pl.DataFrame(
        [
            {
                **base_observation_truncated,
                "zone": None,
            },
        ],
        schema=observation_schema(),
    )
    original_rft_response = pl.DataFrame(
        [
            {**base_original_response_data, "zone": None},
        ],
        schema=original_response_schema(),
    )
    expected_response_df = pl.DataFrame(
        [
            {
                **expected_base_response_data,
            }
        ],
        schema=response_schema(),
    )

    expected_location_metadata_df = pl.DataFrame(
        [
            {
                **expected_base_location_metadata_data,
                "actual_zones": [],
            }
        ],
        schema=location_metadata_schema(),
    )

    response_ds, location_metadata_ds = assert_roundtrip(
        observations_df, original_rft_response
    )
    assert_frame_equal(response_ds, expected_response_df)
    assert_frame_equal(location_metadata_ds, expected_location_metadata_df)


def test_that_migrating_storage_to_28_uses_observations_for_data_on_missing_cell():
    observations_df = pl.DataFrame(
        [
            {**base_observation_truncated},
        ],
        schema=observation_schema(),
    )
    original_rft_response = pl.DataFrame(
        [
            {
                **base_original_response_data,
                "zone": None,
                "east": None,
                "north": None,
                "tvd": None,
            },
        ],
        schema=original_response_schema(),
    )
    expected_response_df = pl.DataFrame(
        [
            {
                **expected_base_response_data,
            }
        ],
        schema=response_schema(),
    )

    expected_location_metadata_df = pl.DataFrame(
        [
            {
                **expected_base_location_metadata_data,
                "actual_zones": [],
                "well_connection_cell": None,
            }
        ],
        schema=location_metadata_schema(),
    )

    response_ds, location_metadata_ds = assert_roundtrip(
        observations_df, original_rft_response
    )
    assert_frame_equal(response_ds, expected_response_df)
    assert_frame_equal(location_metadata_ds, expected_location_metadata_df)


def test_that_migrating_storage_to_28_uses_observations_for_data_on_missing_row():
    observations_df = pl.DataFrame(
        [
            {**base_observation_truncated},
        ],
        schema=observation_schema(),
    )
    original_rft_response = pl.DataFrame([], schema=original_response_schema())
    expected_response_df = pl.DataFrame([], schema=response_schema())

    expected_location_metadata_df = pl.DataFrame(
        [
            {
                **expected_base_location_metadata_data,
                "actual_zones": [],
                "well_connection_cell": None,
            }
        ],
        schema=location_metadata_schema(),
    )

    response_ds, location_metadata_ds = assert_roundtrip(
        observations_df, original_rft_response
    )
    assert_frame_equal(response_ds, expected_response_df)
    assert_frame_equal(location_metadata_ds, expected_location_metadata_df)


def test_that_migrating_storage_to_28_squashes_zones_in_metadata():
    property1 = "PRESSURE"
    property2 = "TEMPERATURE"
    zone1 = "zone1"
    zone2 = "zone2"
    observations_df = pl.DataFrame(
        [
            {**base_observation_truncated, "zone": zone1},
            {**base_observation_truncated, "zone": zone2},
            {**base_observation_truncated, "zone": None},
        ],
        schema=observation_schema(),
    )
    original_rft_response = pl.DataFrame(
        [
            {**base_original_response_data, "zone": zone1, "property": property1},
            {**base_original_response_data, "zone": zone2, "property": property1},
            {**base_original_response_data, "zone": None, "property": property1},
            {**base_original_response_data, "zone": zone1, "property": property2},
            {**base_original_response_data, "zone": zone2, "property": property2},
            {**base_original_response_data, "zone": None, "property": property2},
        ],
        schema=original_response_schema(),
    )
    expected_response_df = pl.DataFrame(
        [
            {**expected_base_response_data, "property": property1},
            {**expected_base_response_data, "property": property2},
        ],
        schema=response_schema(),
    )

    expected_location_metadata_df = pl.DataFrame(
        [
            {
                **expected_base_location_metadata_data,
                "actual_zones": [zone1, zone2],
            }
        ],
        schema=location_metadata_schema(),
    )

    response_ds, location_metadata_ds = assert_roundtrip(
        observations_df, original_rft_response
    )
    assert_frame_equal(response_ds, expected_response_df)
    assert_frame_equal(location_metadata_ds, expected_location_metadata_df)


def test_that_migrating_storage_to_28_catches_well_inconsistency():
    observations_df = pl.DataFrame(
        [
            {
                **base_observation_truncated,
            },
        ],
        schema=observation_schema(),
    )
    original_rft_response = pl.DataFrame(
        [
            {
                **base_original_response_data,
                "i": 10,
                "j": 10,
                "k": 1,
            },
            {
                **base_original_response_data,
                "i": 10,
                "j": 10,
                "k": 2,
            },
        ],
        schema=original_response_schema(),
    )

    msg = "['well', 'date', 'depth']"
    msg += " combinations with inconsistent well_connection_cell values"
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        transform(observations_df, original_rft_response)


def test_that_migrating_storage_to_28_catches_location_inconsistency():
    observations_df = pl.DataFrame(
        [
            {
                **base_observation_truncated,
            },
        ],
        schema=observation_schema(),
    )
    original_rft_response = pl.DataFrame(
        [
            {
                **base_original_response_data,
                "well": "WELL1",
                "i": 10,
                "j": 10,
                "k": 1,
            },
            {
                **base_original_response_data,
                "well": "WELL2",
                "i": 10,
                "j": 10,
                "k": 2,
            },
        ],
        schema=original_response_schema(),
    )

    msg = "['east', 'north', 'tvd']"
    msg += " combinations with inconsistent well_connection_cell values"
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        transform(observations_df, original_rft_response)


def test_that_migrating_storage_to_28_does_not_fail_on_null_location_inconsistency():
    observations_df = pl.DataFrame(
        [
            {
                **base_observation_truncated,
                "observation_key": "OBS1",
                "east": "500.1",
                "north": "1000.1",
                "tvd": "900.1",
                "zone": None,
            },
            {
                **base_observation_truncated,
                "observation_key": "OBS2",
                "east": "600.1",
                "north": "1100.1",
                "tvd": "900.1",
                "zone": "zone1",
            },
        ],
        schema=observation_schema(),
    )
    original_rft_response = pl.DataFrame(
        [
            {
                **base_original_response_data,
                "well": "WELL1",
                "depth": 1000.0,
                "zone": None,
                "east": None,
                "north": None,
                "tvd": None,
                "i": 10,
                "j": 10,
                "k": 1,
            },
            {
                **base_original_response_data,
                "well": "WELL1",
                "depth": 1200.0,
                "zone": None,
                "east": "500.1",
                "north": "1000.1",
                "tvd": "900.1",
                "i": 10,
                "j": 10,
                "k": 2,
            },
            {
                **base_original_response_data,
                "well": "WELL2",
                "depth": 1000.0,
                "zone": None,
                "east": None,
                "north": None,
                "tvd": None,
                "i": 20,
                "j": 20,
                "k": 1,
            },
            {
                **base_original_response_data,
                "well": "WELL2",
                "depth": 1200.0,
                "zone": "zone1",
                "east": "600.1",
                "north": "1100.1",
                "tvd": "900.1",
                "i": 20,
                "j": 20,
                "k": 2,
            },
        ],
        schema=original_response_schema(),
    )

    expected_location_metadata_df = pl.DataFrame(
        [
            {
                **expected_base_location_metadata_data,
                "east": "500.1",
                "north": "1000.1",
                "tvd": "900.1",
                "actual_zones": [],
                "well_connection_cell": [10, 10, 2],
            },
            {
                **expected_base_location_metadata_data,
                "east": "600.1",
                "north": "1100.1",
                "tvd": "900.1",
                "actual_zones": ["zone1"],
                "well_connection_cell": [20, 20, 2],
            },
        ],
        schema=location_metadata_schema(),
    )

    _, location_metadata_ds = assert_roundtrip(observations_df, original_rft_response)
    assert_frame_equal(location_metadata_ds, expected_location_metadata_df)


def test_that_migrating_storage_to_28_splits_large_dataframe():
    date1 = datetime.date(2011, 1, 1)
    date2 = datetime.date(2022, 2, 2)
    response_key1 = f"WELL:{date1}:PRESSURE"
    response_key2 = f"WELL:{date2}:PRESSURE"
    zone1 = "zone1"
    zone2 = "zone2"
    tvd1 = 901.0
    tvd2 = 1000.0
    tvd3 = 902.0
    depth0 = 800.0
    depth1 = 900.0
    depth2 = 1000.0
    value0 = 200.0
    value1 = 300.0
    value2 = 400.0
    ijk0 = [10, 11, 11]
    ijk1 = [10, 11, 12]
    ijk2 = [10, 11, 13]
    observations_df = pl.DataFrame(
        [
            {
                **base_observation_truncated,
                "observation_key": "OBS",
                "response_key": response_key1,
                "zone": zone1,
                "tvd": tvd1,
            },
            {
                **base_observation_truncated,
                "observation_key": "OBS2",
                "response_key": response_key2,
                "zone": zone2,
                "tvd": tvd2,
            },
            {
                **base_observation_truncated,
                "observation_key": "OBS_with_incorrect_zone",
                "response_key": response_key1,
                "zone": zone2,
                "tvd": tvd1,
            },  # disabled
            {
                **base_observation_truncated,
                "observation_key": "OBS_with_slightly_off_tvd",
                "response_key": response_key1,
                "zone": zone1,
                "tvd": tvd3,
            },
            {
                **base_observation_truncated,
                "observation_key": "OBS_with_no_zone",
                "response_key": response_key1,
                "zone": None,
                "tvd": tvd1,
            },
        ],
        schema=observation_schema(),
    )

    def get_response_data(date: datetime.date) -> list[dict[str, Any]]:
        response_key = f"WELL:{date}:PRESSURE"
        return [
            {
                **base_original_response_data,
                "response_key": response_key,
                "time": date,
                "depth": depth0,
                "value": value0,
                "date": date.isoformat(),
                "zone": None,
                "tvd": None,
                "east": None,
                "north": None,
                "i": ijk0[0],
                "j": ijk0[1],
                "k": ijk0[2],
            },
            {
                **base_original_response_data,
                "response_key": response_key,
                "time": date,
                "depth": depth1,
                "value": value1,
                "date": date.isoformat(),
                "zone": zone1,
                "tvd": tvd1,
                "i": ijk1[0],
                "j": ijk1[1],
                "k": ijk1[2],
            },
            {
                **base_original_response_data,
                "response_key": response_key,
                "time": date,
                "depth": depth1,
                "value": value1,
                "date": date.isoformat(),
                "zone": zone1,
                "tvd": tvd3,
                "i": ijk1[0],
                "j": ijk1[1],
                "k": ijk1[2],
            },
            {
                **base_original_response_data,
                "response_key": response_key,
                "time": date,
                "depth": depth1,
                "value": value1,
                "date": date.isoformat(),
                "zone": None,
                "tvd": tvd1,
                "i": ijk1[0],
                "j": ijk1[1],
                "k": ijk1[2],
            },
            {
                **base_original_response_data,
                "response_key": response_key,
                "time": date,
                "depth": depth2,
                "value": value2,
                "date": date.isoformat(),
                "zone": zone2,
                "tvd": tvd2,
                "i": ijk2[0],
                "j": ijk2[1],
                "k": ijk2[2],
            },
        ]

    original_rft_response = pl.DataFrame(
        [*get_response_data(date1), *get_response_data(date2)],
        schema=original_response_schema(),
    )
    assert_roundtrip(observations_df, original_rft_response)
