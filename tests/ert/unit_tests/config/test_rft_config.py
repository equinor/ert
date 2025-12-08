from io import BytesIO

import numpy as np
import pytest
import resfo

from ert.config import (
    InvalidResponseFile,
    RFTConfig,
)
from tests.ert.rft_generator import cell_start, float_arr

original_open = open


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

    mocker.patch("builtins.open", mock_open)

    yield mocked_files


@pytest.fixture
def mock_resfo_file(mocked_files):
    def inner(filename, contents):
        buffer = BytesIO()
        resfo.write(buffer, contents)
        buffer.seek(0)
        mocked_files[filename] = buffer

    return inner


def test_that_rft_with_no_matching_well_and_dates_returns_empty_frame(mock_resfo_file):
    mock_resfo_file("/tmp/does_not_exist/BASE.RFT", [])
    rft_config = RFTConfig(input_files=["BASE.RFT"], data_to_read={})
    assert rft_config.read_from_file("/tmp/does_not_exist", 1, 1).is_empty()


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
    return np.pad(lst, (0, target_len - len(lst)), mode="constant")


def test_that_locations_are_found(mock_resfo_file):
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

    mock_resfo_file(
        "/tmp/does_not_exist/BASE.EGRID",
        [
            ("FILEHEAD", pad_to([3, 2007, 0, 0, 0, 0, 1], 100)),
            ("MAPAXES ", np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=">f4")),
            ("GRIDUNIT", np.array([b"METRES  ", b"        "], dtype="|S8")),
            ("GRIDHEAD", pad_to([1, 2, 2, 2], 100)),
            ("COORD   ", coord.ravel()),
            ("ZCORN   ", np.array([0.0] * 16 + [50] * 32 + [100] * 16, dtype=">f4")),
            ("ACTNUM  ", np.ones((8,), dtype=">i4")),
            ("ENDGRID ", np.array([], dtype=">i4")),
        ],
    )
    mock_resfo_file(
        "/tmp/does_not_exist/BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL2", ijks=[(0, 0, 0)]),
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
    assert data["location"].to_list() == [[1.0, 1.0, 1.0]]
