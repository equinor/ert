import os
from pathlib import Path

import polars as pl
import pytest

from ert.storage.migration.to24 import migrate

original_open = open
original_stat = os.stat
original_scandir = os.scandir


@pytest.fixture
def mocked_files(mocker):
    mocked_files = {}

    def mock_open(*args, **kwargs):
        nonlocal mocked_files
        path = args[0] if args else kwargs.get("file")
        buffer = mocked_files.get(str(path))
        if buffer is not None:
            buffer.seek(0)
            return buffer
        else:
            return original_open(*args, **kwargs)

    def mock_stat(*args, **kwargs):
        nonlocal mocked_files
        path = args[0] if args else kwargs.get("path")
        if str(path) in mocked_files:
            return os.stat_result([0x777, *([1] * 10)])
        else:
            return original_stat(*args, **kwargs)

    mocker.patch("builtins.open", mock_open)
    mocker.patch("os.stat", mock_stat)

    return mocked_files


def mocked_dirs(mocker):
    mocked_dirs = {}

    def mock_scandir(*args, **kwargs):
        nonlocal mocked_dirs
        d = args[0] if args else kwargs["path"]
        if d in mocked_dirs:
            return mocked_dirs[d]
        return original_scandir(*args, **kwargs)

    mocker.patch("os.scandir", mock_scandir)

    return mocked_files


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
