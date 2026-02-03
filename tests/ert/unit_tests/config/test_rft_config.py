import os
from io import BytesIO

import numpy as np
import polars as pl
import pytest
import resfo

from ert.config import (
    ErtConfig,
    InvalidResponseFile,
    RFTConfig,
)
from ert.config._create_observation_dataframes import _handle_rft_observation
from ert.config._observations import RFTObservation
from ert.config.parsing import ObservationType
from ert.warnings import PostSimulationWarning
from tests.ert.rft_generator import cell_start, float_arr

original_open = open
original_stat = os.stat


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


@pytest.fixture
def mock_resfo_file(mocked_files):
    def inner(filename, contents):
        buffer = BytesIO()
        resfo.write(buffer, contents)
        buffer.seek(0)
        mocked_files[filename] = buffer

    return inner


def test_that_rfts_primary_key_is_east_north_tvd_and_zone():
    assert set(
        RFTConfig(
            input_files=["BASE.RFT"],
            data_to_read={"*": {"*": ["*"]}},
        ).primary_key
    ) == {"east", "north", "tvd", "zone"}


def test_that_rft_with_no_matching_well_and_dates_returns_empty_frame(mock_resfo_file):
    mock_resfo_file("/tmp/does_not_exist/BASE.RFT", [])
    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"No such well": {"2000-01-01": ["PRESSURE", "SWAT"]}},
    )
    df = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)

    assert df.is_empty()
    assert set(df.columns) == {
        "response_key",
        "time",
        "depth",
        "values",
        "east",
        "north",
        "tvd",
        "zone",
    }


def test_that_rft_with_empty_data_to_read_returns_empty_df(mock_resfo_file):
    mock_resfo_file("/tmp/does_not_exist/BASE.RFT", [])
    rft_config = RFTConfig(input_files=["BASE.RFT"], data_to_read={})
    df = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)

    assert df.is_empty()
    assert set(df.columns) == {
        "response_key",
        "time",
        "depth",
        "values",
        "east",
        "north",
        "tvd",
        "zone",
    }


def test_that_rft_reads_matching_well_and_date(mock_resfo_file):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL"),
            ("PRESSURE", float_arr([100.0, 200.0])),
            ("SWAT    ", float_arr([0.1, 0.2])),
            ("SGAS    ", float_arr([0.3, 0.4])),
            ("DEPTH   ", float_arr([20.0, 30.0])),
            *cell_start(date=(1, 2, 2000), well_name="WELL"),
            ("PRESSURE", float_arr([101.0, 201.0])),
            ("SWAT    ", float_arr([0.01, 0.01])),
            ("SGAS    ", float_arr([0.01, 0.01])),
            ("DEPTH   ", float_arr([21.0, 31.0])),
        ],
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"WELL": {"2000-01-01": ["PRESSURE", "SWAT"]}},
    )
    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)
    assert data["response_key"].to_list() == [
        "WELL:2000-01-01:PRESSURE",
        "WELL:2000-01-01:PRESSURE",
        "WELL:2000-01-01:SWAT",
        "WELL:2000-01-01:SWAT",
    ]


def test_that_rft_config_can_use_wildcard(mock_resfo_file):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL"),
            ("PRESSURE", float_arr([100.0, 200.0])),
            ("SWAT    ", float_arr([0.1, 0.2])),
            ("SGAS    ", float_arr([0.3, 0.4])),
            ("DEPTH   ", float_arr([20.0, 30.0])),
            *cell_start(date=(1, 2, 2000), well_name="WELL"),
            ("PRESSURE", float_arr([101.0, 201.0])),
            ("SWAT    ", float_arr([0.01, 0.01])),
            ("SGAS    ", float_arr([0.01, 0.01])),
            ("DEPTH   ", float_arr([21.0, 31.0])),
        ],
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["*"]}},
    )
    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)
    assert data["response_key"].to_list() == [
        "WELL:2000-01-01:PRESSURE",
        "WELL:2000-01-01:PRESSURE",
        "WELL:2000-01-01:SWAT",
        "WELL:2000-01-01:SWAT",
        "WELL:2000-01-01:SGAS",
        "WELL:2000-01-01:SGAS",
        "WELL:2000-02-01:PRESSURE",
        "WELL:2000-02-01:PRESSURE",
        "WELL:2000-02-01:SWAT",
        "WELL:2000-02-01:SWAT",
        "WELL:2000-02-01:SGAS",
        "WELL:2000-02-01:SGAS",
    ]


def test_that_reading_from_non_existing_file_raises_invalid_response(tmp_path):
    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"WELL": {"2000-1-1": ["PRESSURE", "SWAT"]}},
    )
    with pytest.raises(InvalidResponseFile):
        rft_config.read_from_file(tmp_path / "BASE.RFT", 1, 1)


def test_that_only_float_arrays_are_read(mock_resfo_file):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL"),
            ("PRESSURE", float_arr([100.0, 200.0])),
            ("HOSTGRID", ["        ", "        "]),
            ("DEPTH   ", float_arr([20.0, 30.0])),
        ],
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["*"]}},
    )
    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)
    assert data["response_key"].to_list() == ["WELL:2000-01-01:PRESSURE"] * 2


def test_that_a_well_can_match_no_properties(mock_resfo_file):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL"),
            ("PRESSURE", float_arr([100.0, 200.0])),
            ("DEPTH   ", float_arr([20.0, 30.0])),
            *cell_start(date=(1, 1, 2000), well_name="WELL2"),
            ("SWAT    ", float_arr([0.0, 0.1])),
            ("DEPTH   ", float_arr([20.0, 30.0])),
        ],
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["SWAT"]}},
    )
    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)
    assert data["response_key"].to_list() == ["WELL2:2000-01-01:SWAT"] * 2


def test_that_missing_depth_raises_invalid_response_file(mock_resfo_file):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL"),
            ("PRESSURE", float_arr([100.0, 200.0])),
        ],
    )
    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["*"]}},
    )
    with pytest.raises(InvalidResponseFile):
        rft_config.read_from_file("/tmp/does_not_exist", 1, 1)


def test_that_number_of_connections_can_be_different_per_well(mock_resfo_file):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL", ijks=[]),
            ("PRESSURE", float_arr([])),
            ("DEPTH   ", float_arr([])),
            *cell_start(date=(1, 1, 2000), well_name="WELL2", ijks=[(0, 0, 0)]),
            ("PRESSURE", float_arr([0.1])),
            ("DEPTH   ", float_arr([0.1])),
        ],
    )
    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["*"]}},
    )
    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)
    assert data["response_key"].to_list() == [
        "WELL2:2000-01-01:PRESSURE",
    ]


def pad_to(lst: list[int], target_len: int):
    return np.pad(
        np.array(lst, dtype=np.int32), (0, target_len - len(lst)), mode="constant"
    )


@pytest.fixture
def egrid():
    """EGrid file contents with three layers.

    The grid is regular with DX and DY = 50, 2 cells
    per direction.

    First layer depth is from 0.0 to 1.0
    Second layer depth is from 1.0 to 2.0
    Third layer depth is from 2.0 to 3.0

    """
    coord = np.array(
        [
            [0, 0, 0, 0, 0, 100],
            [50, 0, 0, 50, 0, 100],
            [100, 0, 0, 100, 0, 100],
            [0, 50, 0, 0, 50, 100],
            [50, 50, 0, 50, 50, 100],
            [100, 50, 0, 100, 50, 100],
            [0, 100, 0, 0, 100, 100],
            [50, 100, 0, 50, 100, 100],
            [100, 100, 0, 100, 100, 100],
        ],
        dtype=">f4",
    )
    return [
        ("FILEHEAD", pad_to([3, 2007, 0, 0, 0, 0, 1], 100)),
        ("MAPAXES ", np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=">f4")),
        ("GRIDUNIT", np.array([b"METRES  ", b"        "], dtype="|S8")),
        ("GRIDHEAD", pad_to([1, 2, 2, 3], 100)),
        ("COORD   ", coord.ravel()),
        (
            "ZCORN   ",
            np.array([0.0] * 16 + [1.0] * 32 + [2.0] * 32 + [3.0] * 16, dtype=">f4"),
        ),
        ("ACTNUM  ", np.ones((8,), dtype=">i4")),
        ("ENDGRID ", np.array([], dtype=">i4")),
    ]


def test_that_locations_are_found(mock_resfo_file, egrid):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.EGRID",
        egrid,
    )
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL2", ijks=[(1, 1, 1)]),
            ("PRESSURE", float_arr([0.1])),
            ("DEPTH   ", float_arr([0.1])),
        ],
    )
    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["*"]}},
        locations=[(1.0, 1.0, 1.0)],
    )
    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)
    assert data["response_key"].to_list() == [
        "WELL2:2000-01-01:PRESSURE",
    ]
    assert data["north"].to_list() == [1.0]
    assert data["east"].to_list() == [1.0]
    assert data["tvd"].to_list() == [1.0]


def test_that_multiple_locations_in_the_same_cell_creates_multiple_rows(
    mock_resfo_file, egrid
):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.EGRID",
        egrid,
    )
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL2", ijks=[(1, 1, 2)]),
            ("PRESSURE", float_arr([0.1])),
            ("DEPTH   ", float_arr([0.1])),
        ],
    )
    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["*"]}},
        locations=[(1.25, 1.25, 1.25), (1.5, 1.5, 1.5)],
    )
    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)
    assert data["response_key"].to_list() == [
        "WELL2:2000-01-01:PRESSURE",
        "WELL2:2000-01-01:PRESSURE",
    ]
    assert sorted(data["north"].to_list()) == [1.25, 1.5]
    assert sorted(data["east"].to_list()) == [1.25, 1.5]
    assert sorted(data["tvd"].to_list()) == [1.25, 1.5]


def test_that_handle_rft_observations_adds_radius_column_to_dataframe():
    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["*"]}},
        locations=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
    )
    rft_observation = RFTObservation(
        name="NAME[0]",
        well="WELL1",
        date="2013-03-31",
        value=294.0,
        error=10.0,
        property="PRESSURE",
        north=71.0,
        east=30.0,
        tvd=2000.0,
    )
    df = _handle_rft_observation(rft_config, rft_observation)
    assert "radius" in df.columns
    assert df["radius"].to_list() == [None]
    assert df["radius"].dtype == pl.Float32


def test_that_if_an_rft_observation_is_outside_the_zone_then_it_is_deactivated(
    mock_resfo_file, egrid
):
    config = ErtConfig.from_dict(
        {
            "ZONEMAP": ("zonemap.txt", {1: ["zone1"], 200: ["zone2"]}),
            "OBS_CONFIG": (
                "obsconf",
                [
                    {
                        "type": ObservationType.RFT,
                        "name": "NAME",
                        "WELL": "well",
                        "VALUE": "700",
                        "ZONE": "zone2",
                        "ERROR": "0.1",
                        "DATE": "2013-03-31",
                        "PROPERTY": "PRESSURE",
                        "NORTH": 1.0,
                        "EAST": 1.0,
                        "TVD": 1.0,
                    }
                ],
            ),
        }
    )
    mock_resfo_file(
        "/tmp/does_not_exist/ECLBASE1.EGRID",
        egrid,
    )
    mock_resfo_file(
        "/tmp/does_not_exist/ECLBASE1.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL2", ijks=[(1, 1, 1)]),
            ("PRESSURE", float_arr([0.1])),
            ("DEPTH   ", float_arr([0.1])),
        ],
    )
    with pytest.warns(PostSimulationWarning, match="did not match expected zone"):
        config.ensemble_config.response_configs["rft"].read_from_file(
            "/tmp/does_not_exist", 1, 1
        )


@pytest.mark.parametrize(
    ("point", "expected_values"),
    [
        (
            (1.0, 1.0, 0.5),
            [
                (0.0, 1.0, 1.0, 0.5, "zone1"),
                (1.0, None, None, None, None),
                (2.0, None, None, None, None),
            ],
        ),
        (
            (1.0, 1.0, 1.5),
            [
                (0.0, None, None, None, None),
                (1.0, 1.0, 1.0, 1.5, "zone1"),
                (1.0, 1.0, 1.0, 1.5, "zone2"),
                (2.0, None, None, None, None),
            ],
        ),
        (
            (1.0, 1.0, 2.5),
            [
                (0.0, None, None, None, None),
                (1.0, None, None, None, None),
                (2.0, 1.0, 1.0, 2.5, "zone2"),
            ],
        ),
    ],
)
def test_that_zone_is_correctly_disabled_for_oberlapping_zones(
    mock_resfo_file, point, expected_values, egrid
):
    config = ErtConfig.from_dict(
        {
            "ZONEMAP": (
                "zonemap.txt",
                {1: ["zone1"], 2: ["zone1", "zone2"], 3: ["zone2"]},
            ),
            "OBS_CONFIG": (
                "obsconf",
                [
                    {
                        "type": ObservationType.RFT,
                        "name": "NAME",
                        "WELL": "WELL",
                        "VALUE": "700",
                        "ERROR": "0.1",
                        "DATE": "2000-01-01",
                        "PROPERTY": "PRESSURE",
                        "NORTH": point[0],
                        "EAST": point[1],
                        "TVD": point[2],
                        "ZONE": "zone1",
                    },
                    {
                        "type": ObservationType.RFT,
                        "name": "NAME",
                        "WELL": "WELL",
                        "VALUE": "700",
                        "ERROR": "0.1",
                        "DATE": "2000-01-01",
                        "PROPERTY": "PRESSURE",
                        "NORTH": point[0],
                        "EAST": point[1],
                        "TVD": point[2],
                        "ZONE": "zone2",
                    },
                ],
            ),
        }
    )
    mock_resfo_file(
        "/tmp/does_not_exist/ECLBASE1.EGRID",
        egrid,
    )
    mock_resfo_file(
        "/tmp/does_not_exist/ECLBASE1.RFT",
        [
            *cell_start(
                date=(1, 1, 2000),
                well_name="WELL",
                ijks=[(1, 1, 1), (1, 1, 2), (1, 1, 3)],
            ),
            ("PRESSURE", float_arr([0.0, 1.0, 2.0])),
            ("DEPTH   ", float_arr([0.0, 1.0, 2.0])),
        ],
    )
    res = config.ensemble_config.response_configs["rft"].read_from_file(
        "/tmp/does_not_exist", 1, 1
    )
    assert (
        sorted(
            zip(
                res["values"].to_list(),
                res["east"].to_list(),
                res["north"].to_list(),
                res["tvd"].to_list(),
                res["zone"].to_list(),
                strict=True,
            )
        )
        == expected_values
    )
