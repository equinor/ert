import io
import os
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest
import resfo

from ert.config import (
    CircleShapeConfig,
    ErtConfig,
    InvalidResponseFile,
    RFTConfig,
    ShapeRegistry,
)
from ert.config._create_observation_dataframes import _handle_rft_observation
from ert.config._observations import RFTObservation
from ert.config.parsing import ConfigValidationError, ObservationType
from ert.config.rft_config import _get_zonemap, _read_egrid
from tests.ert.rft_generator import cell_start, float_arr

original_open = open
original_io_open = io.open
original_stat = os.stat


@pytest.fixture(autouse=True)
def _clear_rft_caches():
    """Reset the module-level EGRID and zonemap caches so cached entries from
    previous tests do not leak into tests that expect a missing or different file."""
    _read_egrid.cache_clear()
    _get_zonemap.cache_clear()
    yield
    _read_egrid.cache_clear()
    _get_zonemap.cache_clear()


@pytest.fixture
def mocked_files(mocker):
    mocked_files = {}

    def _fresh_buffer(data):
        if isinstance(data, bytes):
            return BytesIO(data)
        return StringIO(data)

    def mock_open(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        data = mocked_files.get(str(path))
        if data is not None:
            return _fresh_buffer(data)
        return original_open(*args, **kwargs)

    def mock_io_open(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        data = mocked_files.get(str(path))
        if data is not None:
            return _fresh_buffer(data)
        return original_io_open(*args, **kwargs)

    def mock_stat(*args, **kwargs):
        path = args[0] if args else kwargs.get("path")
        if str(path) in mocked_files:
            return os.stat_result([0x777, *([1] * 10)])
        return original_stat(*args, **kwargs)

    mocker.patch("builtins.open", mock_open)
    mocker.patch("io.open", mock_io_open)
    mocker.patch("os.stat", mock_stat)

    return mocked_files


@pytest.fixture
def mock_resfo_file(mocked_files):
    def inner(filename, contents):
        buffer = BytesIO()
        resfo.write(buffer, contents)
        mocked_files[filename] = buffer.getvalue()

    return inner


def test_that_rfts_match_key_is_well_connection_cell():
    assert set(
        RFTConfig(
            input_files=["BASE.RFT"],
            data_to_read={"*": {"*": ["*"]}},
        ).match_key
    ) == {"well_connection_cell"}


def test_that_match_key_dict_expr_fits_match_key():
    date = datetime(2000, 1, 1).date()  # noqa: DTZ001

    def response(well_connection_cell) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "response_key": ["WELL:2000-01-01:SWAT"],
                "well": ["WELL"],
                "date": [date.isoformat()],
                "property": ["SWAT"],
                "time": [date],
                "depth": [1006.6],
                "values": [100.0],
                "well_connection_cell": pl.Series(
                    [well_connection_cell], dtype=pl.Array(pl.Int64, 3)
                ),
                "cell_center": [[0.0, 0.0, 0.0]],
                "cell_zones": [[]],
            },
            schema=RFTConfig.response_schema(),
        )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
    )

    response1 = response([10, 11, 12]).with_columns(
        rft_config.match_key_dict_expr().alias("response_dict")
    )
    assert response1["response_dict"].to_list() == ["well_connection_cell=[10, 11, 12]"]

    response2 = response(None).with_columns(
        rft_config.match_key_dict_expr().alias("response_dict")
    )
    assert response2["response_dict"].to_list() == ["well_connection_cell=None"]


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
        "date",
        "well",
        "property",
        "time",
        "depth",
        "values",
        "well_connection_cell",
        "cell_center",
        "cell_zones",
    }


def test_that_rft_with_empty_data_to_read_returns_empty_df(mock_resfo_file):
    mock_resfo_file("/tmp/does_not_exist/BASE.RFT", [])
    rft_config = RFTConfig(input_files=["BASE.RFT"], data_to_read={})
    df = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)

    assert df.is_empty()
    assert set(df.columns) == {
        "response_key",
        "date",
        "well",
        "property",
        "time",
        "depth",
        "values",
        "well_connection_cell",
        "cell_center",
        "cell_zones",
    }


def test_that_rft_reads_matching_well_and_date(mock_resfo_file, egrid):
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
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.EGRID",
        egrid,
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


def test_that_rft_config_wildcards_matches_any_well_date_and_property(
    mock_resfo_file, egrid
):
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
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.EGRID",
        egrid,
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


def test_that_missing_egrid_file_raises_invalid_response(mock_resfo_file):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL"),
            ("PRESSURE", float_arr([100.0, 200.0])),
            ("DEPTH   ", float_arr([20.0, 30.0])),
        ],
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["*"]}},
    )
    with pytest.raises(InvalidResponseFile):
        rft_config.read_from_file("/tmp/does_not_exist", 1, 1)


def test_that_only_float_arrays_are_read(mock_resfo_file, egrid):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL"),
            ("PRESSURE", float_arr([100.0, 200.0])),
            ("HOSTGRID", ["        ", "        "]),
            ("DEPTH   ", float_arr([20.0, 30.0])),
        ],
    )
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.EGRID",
        egrid,
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["*"]}},
    )
    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)
    assert data["response_key"].to_list() == ["WELL:2000-01-01:PRESSURE"] * 2


def test_that_a_well_can_match_no_properties(mock_resfo_file, egrid):
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
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.EGRID",
        egrid,
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["SWAT"]}},
    )
    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)
    assert data["response_key"].to_list() == ["WELL2:2000-01-01:SWAT"] * 2


def test_that_missing_depth_raises_invalid_response_file(mock_resfo_file, egrid):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL"),
            ("PRESSURE", float_arr([100.0, 200.0])),
        ],
    )
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.EGRID",
        egrid,
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["*"]}},
    )
    with pytest.raises(InvalidResponseFile):
        rft_config.read_from_file("/tmp/does_not_exist", 1, 1)


def test_that_too_few_property_values_raises_invalid_response_file(mock_resfo_file):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(
                date=(1, 1, 2000),
                well_name="WELL1",
                ijks=((1, 1, 1), (2, 2, 2)),  # two connections
            ),
            ("PRESSURE", float_arr([100.0])),  # only one pressure value
            ("DEPTH   ", float_arr([20.0])),
        ],
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"2000-01-01": ["PRESSURE"]}},
    )
    with pytest.raises(InvalidResponseFile, match="has 1 value but 2 well connections"):
        rft_config.read_from_file("/tmp/does_not_exist", 1, 1)


def test_that_number_of_connections_can_be_different_per_well(mock_resfo_file, egrid):
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
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.EGRID",
        egrid,
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


def _egrid(nx, ny, nz, x_width, y_width, layer_height):
    """EGrid file contents with nz layers, nx cells in the i direction and ny cells in
    the j direction.

    Each cell has width x_width in the i direction and y_width in the j direction and
    height layer_height in the z direction.
    """

    height = nz * layer_height
    cells_per_layer = nx * ny

    coord = np.array(
        [
            [i * x_width, j * y_width, 0, i * x_width, j * y_width, height]
            for j in range(ny + 1)
            for i in range(nx + 1)
        ],
        dtype=">f4",
    )

    zcoord = np.array(
        [
            [z * layer_height] * (cells_per_layer * 4)
            + [(z + 1) * layer_height] * (cells_per_layer * 4)
            for z in range(nz)
        ],
        dtype=">f4",
    )
    return [
        ("FILEHEAD", pad_to([3, 2007, 0, 0, 0, 0, 1], 100)),
        ("MAPAXES ", np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=">f4")),
        ("GRIDUNIT", np.array([b"METRES  ", b"        "], dtype="|S8")),
        ("GRIDHEAD", pad_to([1, nx, ny, nz], 100)),
        ("COORD   ", coord.ravel()),
        ("ZCORN   ", zcoord.ravel()),
        ("ACTNUM  ", np.ones(nx * ny * nz, dtype=">i4")),
        ("ENDGRID ", np.array([], dtype=">i4")),
    ]


@pytest.fixture
def egrid():
    """EGrid file contents with three layers.

    The grid is regular with DX and DY = 50, 2 cells
    per direction.

    First layer depth is from 0.0 to 1.0
    Second layer depth is from 1.0 to 2.0
    Third layer depth is from 2.0 to 3.0

    """
    return _egrid(2, 2, 3, 50.0, 50.0, 1.0)


def test_that_locations_are_found_in_corresponding_grid(mock_resfo_file, egrid):
    config = ErtConfig.from_dict(
        {
            "ECLBASE": "BASE",
            "OBS_CONFIG": (
                "obsconf",
                [
                    {
                        "type": ObservationType.RFT,
                        "name": "OBS1",
                        "WELL": "WELL2",
                        "VALUE": "7",
                        "ERROR": "0.1",
                        "DATE": "2000-01-01",
                        "PROPERTY": "PRESSURE",
                        "NORTH": 1.0,
                        "EAST": 1.0,
                        "TVD": 1.0,
                    },
                ],
            ),
        }
    )

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
    rft_config = cast(RFTConfig, config.ensemble_config.response_configs["rft"])
    observations = pl.DataFrame(config.observation_declarations)
    data = rft_config.obtain_location_metadata(
        "/tmp/does_not_exist", 1, 1, observations
    )

    assert data["north"].to_list() == [1.0]
    assert data["east"].to_list() == [1.0]
    assert data["tvd"].to_list() == [1.0]


def test_that_multiple_locations_in_the_same_cell_creates_multiple_rows(
    mock_resfo_file, egrid
):
    config = ErtConfig.from_dict(
        {
            "ECLBASE": "BASE",
            "OBS_CONFIG": (
                "obsconf",
                [
                    {
                        "type": ObservationType.RFT,
                        "name": "OBS1",
                        "WELL": "WELL2",
                        "VALUE": "7",
                        "ERROR": "0.1",
                        "DATE": "2000-01-01",
                        "PROPERTY": "PRESSURE",
                        "NORTH": 1.25,
                        "EAST": 1.25,
                        "TVD": 1.25,
                    },
                    {
                        "type": ObservationType.RFT,
                        "name": "OBS2",
                        "WELL": "WELL2",
                        "VALUE": "8",
                        "ERROR": "0.1",
                        "DATE": "2000-01-01",
                        "PROPERTY": "PRESSURE",
                        "NORTH": 1.5,
                        "EAST": 1.5,
                        "TVD": 1.5,
                    },
                ],
            ),
        }
    )

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

    rft_config = cast(RFTConfig, config.ensemble_config.response_configs["rft"])
    observations = pl.DataFrame(config.observation_declarations)
    data = rft_config.obtain_location_metadata(
        "/tmp/does_not_exist", 1, 1, observations
    )
    assert sorted(data["north"].to_list()) == [1.25, 1.5]
    assert sorted(data["east"].to_list()) == [1.25, 1.5]
    assert sorted(data["tvd"].to_list()) == [1.25, 1.5]


def test_that_connection_cells_are_processed_independently(mock_resfo_file):
    config = ErtConfig.from_dict(
        {
            "ECLBASE": "ECLBASE<IENS>",
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
                        "NORTH": 1.0,
                        "EAST": 1.0,
                        "TVD": 1.5,
                    },
                ],
            ),
        }
    )

    mock_resfo_file(
        "/tmp/does_not_exist/ECLBASE1.EGRID",
        _egrid(5, 2, 3, 50.0, 50.0, 1.0),
    )
    mock_resfo_file(
        "/tmp/does_not_exist/ECLBASE1.RFT",
        [
            *cell_start(
                date=(1, 1, 2000),
                well_name="WELL",
                ijks=[(5, 1, 1), (5, 1, 2), (5, 1, 3)],
            ),
            ("PRESSURE", float_arr([0.0, 1.0, 2.0])),
            ("DEPTH   ", float_arr([0.0, 1.0, 2.0])),
        ],
    )

    rft_config = cast(RFTConfig, config.ensemble_config.response_configs["rft"])
    iens = 1
    iter_ = 1
    responses = rft_config.read_from_file("/tmp/does_not_exist", iens, iter_)
    observations = pl.DataFrame(config.observation_declarations)
    observation_metadata = rft_config.obtain_location_metadata(
        "/tmp/does_not_exist", iens, iter_, observations
    )
    assert responses["well_connection_cell"].to_list() == [
        [5, 1, 1],
        [5, 1, 2],
        [5, 1, 3],
    ]
    assert observation_metadata["well_connection_cell"].to_list() == [[1, 1, 2]]
    assert observation_metadata["actual_zones"].to_list() == [[]]


def test_that_location_outside_of_the_grid_raises_invalid_response_file(
    mock_resfo_file, egrid
):
    config = ErtConfig.from_dict(
        {
            "ECLBASE": "ECLBASE<IENS>",
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
                        "NORTH": 1000.0,
                        "EAST": 1000.0,
                        "TVD": 1500.0,
                    },
                ],
            ),
        }
    )

    mock_resfo_file(
        "/tmp/does_not_exist/ECLBASE1.EGRID",
        egrid,
    )

    rft_config = cast(RFTConfig, config.ensemble_config.response_configs["rft"])
    observations = pl.DataFrame(config.observation_declarations)
    with pytest.raises(InvalidResponseFile, match="Did not find grid coordinate"):
        rft_config.obtain_location_metadata("/tmp/does_not_exist", 1, 1, observations)


def test_that_handle_rft_observations_prioritize_provided_radius_over_default():
    provided_radius = 2400
    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["*"]}},
    )
    shape_registry = ShapeRegistry()
    shape_id = shape_registry.register(
        CircleShapeConfig(
            east=30.0,
            north=71.0,
            radius=provided_radius,
        )
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
        shape_id=shape_id,
    )
    df = _handle_rft_observation(rft_config, rft_observation, shape_registry)
    assert "radius" in df.columns
    assert df["radius"].to_list() == [provided_radius]
    assert df["radius"].dtype == pl.Float32


def test_that_when_the_zonemap_is_an_absolute_path_then_the_runpath_is_not_prepended(
    mock_resfo_file, egrid, mocked_files
):
    mocked_files["/tmp/does_not_exist/zonemap.txt"] = "this_is_an_invalid_zonemap zone1"
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

    config = ErtConfig.from_dict(
        {
            "ZONEMAP": "/tmp/does_not_exist/zonemap.txt",
            "ECLBASE": "ECLBASE<IENS>",
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
                        "NORTH": 1.0,
                        "EAST": 1.0,
                        "TVD": 0.5,
                        "ZONE": "zone2",
                    },
                ],
            ),
        }
    )
    rft_config = cast(RFTConfig, config.ensemble_config.response_configs["rft"])
    observations = pl.DataFrame(config.observation_declarations)
    with pytest.raises(
        ConfigValidationError,
        match="must be an integer, was this_is_an_invalid_zonemap",
    ):
        rft_config.obtain_location_metadata("/tmp/does_not_exist", 1, 1, observations)


@pytest.mark.parametrize(
    ("zonemap_path"),
    [
        pytest.param(
            "zonemap<IENS>-<ITER>.txt",
            id="when path is relative",
        ),
        pytest.param(
            "/tmp/does_not_exist/zonemap<IENS>-<ITER>.txt",
            id="when path is absolute",
        ),
    ],
)
def test_that_substitutions_are_applied_to_zonemap_filename(
    mocked_files, mock_resfo_file, egrid, zonemap_path
):
    iens = 5
    iter_ = 2
    mocked_files[f"/tmp/does_not_exist/zonemap{iens}-{iter_}.txt"] = (
        "zonemap_file_is_picked_correctly zone1"
    )

    mock_resfo_file(
        f"/tmp/does_not_exist/ECLBASE{iens}.EGRID",
        egrid,
    )
    mock_resfo_file(
        f"/tmp/does_not_exist/ECLBASE{iens}.RFT",
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

    config = ErtConfig.from_dict(
        {
            "ZONEMAP": zonemap_path,
            "ECLBASE": "ECLBASE<IENS>",
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
                        "NORTH": 1.0,
                        "EAST": 1.0,
                        "TVD": 0.5,
                        "ZONE": "zone1",
                    },
                ],
            ),
        }
    )

    rft_config = cast(RFTConfig, config.ensemble_config.response_configs["rft"])
    observations = pl.DataFrame(config.observation_declarations)
    with pytest.raises(
        ConfigValidationError,
        match="must be an integer, was zonemap_file_is_picked_correctly",
    ):
        rft_config.obtain_location_metadata(
            "/tmp/does_not_exist", iens, iter_, observations
        )


@pytest.mark.filterwarnings("ignore:.*contains a RFT key but no forward model step")
def test_that_missing_egrid_with_locations_raises_invalid_response_file(
    mock_resfo_file,
):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL", ijks=[(1, 1, 1)]),
            ("PRESSURE", float_arr([100.0])),
            ("DEPTH   ", float_arr([20.0])),
        ],
    )

    config = ErtConfig.from_dict(
        {
            "ECLBASE": "BASE.RFT",
            "RFT": [{"WELL": "*", "DATE": "*", "PROPERTIES": "PRESSURE"}],
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
                        "NORTH": 1.0,
                        "EAST": 1.0,
                        "TVD": 1.0,
                    },
                ],
            ),
        }
    )

    rft_config = cast(RFTConfig, config.ensemble_config.response_configs["rft"])
    observations = pl.DataFrame(config.observation_declarations)
    with pytest.raises(InvalidResponseFile):
        rft_config.obtain_location_metadata("/tmp/does_not_exist", 1, 1, observations)


def test_that_missing_egrid_without_locations_raises_invalid_response_file(
    mock_resfo_file,
):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL", ijks=[(1, 1, 1)]),
            ("PRESSURE", float_arr([100.0])),
            ("DEPTH   ", float_arr([20.0])),
        ],
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"*": ["PRESSURE"]}},
    )

    with pytest.raises(InvalidResponseFile):
        rft_config.read_from_file("/tmp/does_not_exist", 1, 1)


def test_that_zone_with_multiple_layers_produces_single_matching_row(
    mock_resfo_file, egrid, mocked_files
):
    mocked_files["/tmp/does_not_exist/zonemap.txt"] = "1 zone1\n2 zone1\n3 zone3\n"

    config = ErtConfig.from_dict(
        {
            "ZONEMAP": "zonemap.txt",
            "ECLBASE": "ECLBASE<IENS>",
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
                        "NORTH": 1.0,
                        "EAST": 1.0,
                        "TVD": 1.5,
                        "ZONE": "zone1",
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
            *cell_start(
                date=(1, 1, 2000),
                well_name="WELL",
                ijks=[(1, 1, 1), (1, 1, 2), (1, 1, 3)],
            ),
            ("PRESSURE", float_arr([0.0, 1.0, 2.0])),
            ("DEPTH   ", float_arr([0.0, 1.0, 2.0])),
        ],
    )

    rft_config = cast(RFTConfig, config.ensemble_config.response_configs["rft"])
    observations = pl.DataFrame(config.observation_declarations)
    observation_metadata = rft_config.obtain_location_metadata(
        "/tmp/does_not_exist", 1, 1, observations
    )

    assert observation_metadata["well_connection_cell"].to_list() == [[1, 1, 2]]
    assert observation_metadata["actual_zones"].to_list() == [["zone1"]]


def test_that_non_matching_wells_are_ignored_silently(mock_resfo_file, egrid):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL_MATCH"),
            ("PRESSURE", float_arr([100.0, 200.0])),
            ("DEPTH   ", float_arr([20.0, 30.0])),
            *cell_start(date=(1, 1, 2000), well_name="WELL_OTHER"),
            ("PRESSURE", float_arr([300.0, 400.0])),
            ("DEPTH   ", float_arr([40.0, 50.0])),
        ],
    )
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.EGRID",
        egrid,
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"WELL_MATCH": {"2000-01-01": ["PRESSURE"]}},
    )

    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)
    assert data["response_key"].to_list() == [
        "WELL_MATCH:2000-01-01:PRESSURE",
        "WELL_MATCH:2000-01-01:PRESSURE",
    ]


def test_that_wildcard_well_with_specific_date_matches_all_wells(
    mock_resfo_file, egrid
):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL1", ijks=((1, 1, 1),)),
            ("PRESSURE", float_arr([100.0])),
            ("DEPTH   ", float_arr([20.0])),
            *cell_start(date=(1, 2, 2000), well_name="WELL1", ijks=((1, 1, 1),)),
            ("PRESSURE", float_arr([101.0])),
            ("DEPTH   ", float_arr([21.0])),
            *cell_start(date=(1, 1, 2000), well_name="WELL2", ijks=((1, 1, 1),)),
            ("PRESSURE", float_arr([200.0])),
            ("DEPTH   ", float_arr([30.0])),
        ],
    )
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.EGRID",
        egrid,
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"*": {"2000-01-01": ["PRESSURE"]}},
    )

    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)
    assert data["response_key"].to_list() == [
        "WELL1:2000-01-01:PRESSURE",
        "WELL2:2000-01-01:PRESSURE",
    ]


def test_that_specific_well_with_wildcard_property_reads_all_float_arrays(
    mock_resfo_file, egrid
):
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL", ijks=((1, 1, 1),)),
            ("PRESSURE", float_arr([100.0])),
            ("SWAT    ", float_arr([0.1])),
            ("SGAS    ", float_arr([0.2])),
            ("HOSTGRID", ["        "]),
            ("DEPTH   ", float_arr([20.0])),
        ],
    )
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.EGRID",
        egrid,
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"WELL": {"2000-01-01": ["*"]}},
    )

    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)
    assert sorted(data["response_key"].to_list()) == sorted(
        [
            "WELL:2000-01-01:PRESSURE",
            "WELL:2000-01-01:SWAT",
            "WELL:2000-01-01:SGAS",
        ]
    )


@pytest.mark.parametrize(
    ("with_zonemap", "expected_zones"),
    [
        pytest.param(False, [None, None]),
        pytest.param(True, [["zone_a"], ["zone_b"]]),
    ],
)
def test_that_cell_center_and_cell_zones_are_populated_from_egrid_and_zonemap(
    mock_resfo_file,
    egrid,
    mocked_files,
    with_zonemap,
    expected_zones,
):
    mocked_files["/tmp/does_not_exist/zonemap.txt"] = "1 zone_a\n2 zone_b\n"
    mock_resfo_file("/tmp/does_not_exist/BASE.EGRID", egrid)
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(
                date=(1, 1, 2000),
                well_name=b"WELL",
                ijks=[(1, 1, 1), (1, 1, 2)],
            ),
            ("PRESSURE", float_arr([10.0, 20.0])),
            ("DEPTH   ", float_arr([0.5, 1.5])),
        ],
    )

    rft_config = RFTConfig(
        input_files=["BASE.RFT"],
        data_to_read={"WELL": {"2000-01-01": ["PRESSURE"]}},
        zonemap=Path("zonemap.txt") if with_zonemap else None,
    )

    data = rft_config.read_from_file("/tmp/does_not_exist", 1, 1)

    assert data["well_connection_cell"].to_list() == [[1, 1, 1], [1, 1, 2]]
    assert data["cell_zones"].to_list() == expected_zones
    np.testing.assert_allclose(
        data["cell_center"].to_numpy(), [[25.0, 25.0, 0.5], [25.0, 25.0, 1.5]]
    )
