import os
from pathlib import Path

import polars as pl
import pytest

from ert.storage.migration.to24 import migrate


@pytest.mark.usefixtures("use_tmpdir")
def test_that_migrating_storage_to_24_adds_zone_keyword_to_responses():
    os.makedirs("ensembles/hash/realization-0")

    pl.DataFrame(
        {
            "response_key": [],
            "time": [],
            "depth": [],
            "values": [],
            "east": [],
            "north": [],
            "tvd": [],
        }
    ).write_parquet("ensembles/hash/realization-0/rft.parquet")
    migrate(Path("."))

    assert "zone" in pl.read_parquet("ensembles/hash/realization-0/rft.parquet").columns


@pytest.mark.usefixtures("use_tmpdir")
def test_that_migrating_storage_to_24_adds_zone_keyword_to_observations():
    os.makedirs("experiments/hash/observations")

    pl.DataFrame(
        {
            "response_key": [],
            "time": [],
            "depth": [],
            "values": [],
            "east": [],
            "north": [],
            "tvd": [],
        }
    ).write_parquet("experiments/hash/observations/rft")
    migrate(Path("."))

    assert "zone" in pl.read_parquet("experiments/hash/observations/rft").columns
