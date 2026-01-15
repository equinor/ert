import os
from contextlib import suppress
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given

from ert.config import (
    ConfigValidationError,
    ConfigWarning,
    ErtConfig,
    GenDataConfig,
    InvalidResponseFile,
)


@pytest.mark.parametrize(
    ("name", "report_steps"),
    [
        ("ORDERED_RESULTS", [1, 2, 3, 4]),
        ("UNORDERED_RESULTS", [5, 2, 3, 7, 1]),
    ],
)
def test_report_step_list_is_ordered(name: str, report_steps: list[int]):
    gdc = GenDataConfig(keys=[name], report_steps_list=[report_steps])
    assert gdc.keys == [name]
    assert gdc.report_steps_list[0] == sorted(report_steps)


def test_that_filter_on_property_maps_report_steps_to_each_key():
    report_steps_0 = [0, 3, 6, 9]
    report_steps_1 = [1, 4, 7, 10]
    report_steps_2 = [2, 5, 8, 11]
    gdc = GenDataConfig(
        keys=["key_0", "key_1", "key_2"],
        report_steps_list=[report_steps_0, report_steps_1, report_steps_2],
    )
    assert gdc.filter_on["key_0"]["report_step"] == sorted(report_steps_0)
    assert gdc.filter_on["key_1"]["report_step"] == sorted(report_steps_1)
    assert gdc.filter_on["key_2"]["report_step"] == sorted(report_steps_2)


def test_gen_data_default_report_step():
    gen_data_default_step = GenDataConfig(keys=["name"])
    assert not gen_data_default_step.report_steps_list[0][0]


def test_empty_result_file_gives_validation_error():
    with pytest.raises(ConfigValidationError, match="Missing RESULT_FILE for GEN_DATA"):
        ErtConfig.from_file_contents("NUM_REALIZATIONS 1\nGEN_DATA NAME")


def test_unset_result_file_gives_validation_error():
    with pytest.raises(ConfigValidationError, match="Missing RESULT_FILE for GEN_DATA"):
        GenDataConfig.from_config_dict({"GEN_DATA": [["NAME", {}]]})


def test_invalid_result_file_gives_validation_error():
    with pytest.raises(ConfigValidationError, match=r"RESULT_FILE:/tmp .* is invalid"):
        GenDataConfig.from_config_dict(
            {"GEN_DATA": [["NAME", {"RESULT_FILE": "/tmp"}]]}
        )


def test_result_file_is_appended_to_input_files():
    gen_data = GenDataConfig.from_config_dict(
        {"GEN_DATA": [["NAME", {"RESULT_FILE": "%d.out", "REPORT_STEPS": "1"}]]}
    )
    assert "%d.out" in gen_data.input_files


def test_that_gen_data_input_format_yields_deprecation_warning():
    with pytest.warns(
        ConfigWarning,
        match="INPUT_FORMAT has been removed since 2023, and has no effect.",
    ):
        GenDataConfig.from_config_dict(
            {
                "GEN_DATA": [
                    [
                        "NAME",
                        {
                            "INPUT_FORMAT": "ASCII",
                            "RESULT_FILE": "%d",
                            "REPORT_STEPS": "1-2,5-8",
                        },
                    ]
                ]
            }
        )


@pytest.mark.parametrize("not_a_range", ["H", "H,1-3", "invalid-range-argument"])
def test_non_range_report_step_gives_validation_error(not_a_range):
    with pytest.raises(ConfigValidationError, match="must be a valid range string"):
        GenDataConfig.from_config_dict(
            {"GEN_DATA": [["NAME", {"RESULT_FILE": "%d", "REPORT_STEPS": not_a_range}]]}
        )


def test_report_step_option_sets_the_report_steps_property():
    gen_data = GenDataConfig.from_config_dict(
        {"GEN_DATA": [["NAME", {"RESULT_FILE": "%d", "REPORT_STEPS": "1-2,5-8"}]]}
    )
    assert gen_data.report_steps_list == [[1, 2, 5, 6, 7, 8]]


def test_that_invalid_gendata_outfile_error_propagates(tmp_path):
    (tmp_path / "poly.out").write_text("""
        4.910405046410615,4.910405046410615
        6.562317389289953,6.562317389289953
        9.599763191512997,9.599763191512997
        14.022742453079745,14.022742453079745
        19.831255173990197,19.831255173990197
        27.025301354244355,27.025301354244355
        35.604880993842215,35.604880993842215
        45.56999409278378,45.56999409278378
        56.92064065106905,56.92064065106905
        69.65682066869802,69.65682066869802
    """)

    config = GenDataConfig(
        name="gen_data",
        keys=["something"],
        report_steps_list=[None],
        input_files=["poly.out"],
    )
    with pytest.raises(
        InvalidResponseFile,
        match=(
            r"Error reading GEN_DATA.*could not convert string."
            r"*4.910405046410615,4.910405046410615.*to float64"
        ),
    ):
        config.read_from_file(tmp_path, 0, 0)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.filterwarnings("ignore:.*loadtxt.*input contained no data.*")
@given(st.binary())
def test_that_read_file_does_not_raise_unexpected_exceptions_on_invalid_file(contents):
    Path("./output").write_bytes(contents)
    with suppress(InvalidResponseFile):
        GenDataConfig(
            name="gen_data",
            keys=["something"],
            report_steps_list=[None],
            input_files=["output"],
        ).read_from_file(os.getcwd(), 0, 0)


def test_that_read_file_does_not_raise_unexpected_exceptions_on_missing_file(tmpdir):
    with pytest.raises(FileNotFoundError, match="DOES_NOT_EXIST not found"):
        GenDataConfig(
            name="gen_data",
            keys=["something"],
            report_steps_list=[None],
            input_files=["DOES_NOT_EXIST"],
        ).read_from_file(tmpdir, 0, 0)


def test_that_read_file_does_not_raise_unexpected_exceptions_on_missing_directory(
    tmp_path,
):
    with pytest.raises(FileNotFoundError, match="DOES_NOT_EXIST not found"):
        GenDataConfig(
            name="gen_data",
            keys=["something"],
            report_steps_list=[None],
            input_files=["DOES_NOT_EXIST"],
        ).read_from_file(str(tmp_path / "DOES_NOT_EXIST"), 0, 0)
