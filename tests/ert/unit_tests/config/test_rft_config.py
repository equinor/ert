from functools import partial
from io import BytesIO

import numpy as np
import pytest
import resfo

from ert.config import (
    InvalidResponseFile,
    RFTConfig,
)


def well_etc(
    time_units=b"HOURS",
    lgr_name=b"",
    data_category=b"R",
    well_name=b"WELL1",
):
    return np.array(
        [
            time_units,
            well_name,
            lgr_name,
            b"METRES",
            b"BARSA",
            data_category,
            b"STANDARD",
            b"SM3/DAY",
            b"SM3/DAY",
            b"RM3/DAY",
            b"",
            b"M/SEC",
            b"CP",
            b"KG/SM3",
            b"KG/DAY",
            b"KG/KG",
        ]
    )


float_arr = partial(np.array, dtype=np.float32)
int_arr = partial(np.array, dtype=np.int32)


def cell_start(date=(1, 1, 2000), ijks=((1, 1, 1), (2, 1, 2)), *args, **kwargs):
    return [
        ("TIME    ", float_arr([24.0])),
        ("DATE    ", int_arr(date)),
        ("WELLETC ", well_etc(*args, **kwargs)),
        ("CONIPOS ", int_arr([i for i, _, _ in ijks])),
        ("CONJPOS ", int_arr([j for _, j, _ in ijks])),
        ("CONKPOS ", int_arr([k for _, _, k in ijks])),
    ]


def plt_fields():
    return [
        ("CONWRAT ", float_arr([2.0, 3.0])),
        ("CONGRAT ", float_arr([2.0, 3.0])),
        ("CONORAT ", float_arr([2.0, 3.0])),
        ("CONDEPTH", float_arr([2.0, 3.0])),
        ("CONVTUB ", float_arr([2.0, 3.0])),
        ("CONOTUB ", float_arr([2.0, 3.0])),
        ("CONGTUB ", float_arr([2.0, 3.0])),
        ("CONWTUB ", float_arr([2.0, 3.0])),
        ("CONPRES ", float_arr([0.0, 0.0])),
    ]


original_open = open


def mock_rft_file(mocker, filename, contents):
    buffer = BytesIO()
    resfo.write(buffer, contents)
    buffer.seek(0)

    def mock_open(*args, **kwargs):
        path = args[0] if args else kwargs.get("file")
        if path is not None and str(path).endswith(filename):
            return buffer
        else:
            return original_open(*args, **kwargs)

    mocker.patch("builtins.open", mock_open)


def test_that_rft_with_no_matching_well_and_dates_returns_empty_frame(mocker):
    mock_rft_file(mocker, "BASE.RFT", [])
    rft_config = RFTConfig(input_files=["BASE.RFT"], data_to_read={})
    assert rft_config.read_from_file("/tmp/does_not_exist", 1, 1).is_empty()


def test_that_rft_reads_matching_well_and_date(mocker):
    mock_rft_file(
        mocker,
        "BASE.RFT",
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


def test_that_rft_config_can_use_wildcard(mocker):
    mock_rft_file(
        mocker,
        "BASE.RFT",
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


def test_that_only_float_arrays_are_read(mocker):
    mock_rft_file(
        mocker,
        "BASE.RFT",
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


def test_that_a_well_can_match_no_properties(mocker):
    mock_rft_file(
        mocker,
        "BASE.RFT",
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


def test_that_missing_depth_raises_invalid_response_file(mocker):
    mock_rft_file(
        mocker,
        "BASE.RFT",
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


def test_that_number_of_connections_can_be_different_per_well(mocker):
    mock_rft_file(
        mocker,
        "BASE.RFT",
        [
            *cell_start(date=(1, 1, 2000), well_name="WELL"),
            ("PRESSURE", float_arr([])),
            ("DEPTH   ", float_arr([])),
            *cell_start(date=(1, 1, 2000), well_name="WELL2"),
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
