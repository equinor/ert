from typing import List

import pytest

from ert.config import ConfigValidationError, GenDataConfig


@pytest.mark.parametrize(
    "name, report_steps",
    [
        ("ORDERED_RESULTS", [1, 2, 3, 4]),
        ("UNORDERED_RESULTS", [5, 2, 3, 7, 1]),
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_gen_data_config(name: str, report_steps: List[int]):
    gdc = GenDataConfig(keys=[name], report_steps_list=[report_steps])
    assert gdc.keys == [name]
    assert gdc.report_steps_list[0] == sorted(report_steps)


def test_gen_data_default_report_step():
    gen_data_default_step = GenDataConfig(keys=["name"])
    assert not gen_data_default_step.report_steps_list[0][0]


@pytest.mark.parametrize(
    "result_file, error_message",
    [
        pytest.param(
            "RESULT_FILE:",
            "Invalid argument 'RESULT_FILE:'",
            id="RESULT_FILE key but no file",
        ),
        pytest.param(
            "",
            "Missing or unsupported RESULT_FILE for GEN_DATA",
            id="No RESULT_FILE key",
        ),
        pytest.param(
            "RESULT_FILE:/tmp",
            "The RESULT_FILE:/tmp setting for RES is invalid",
            id="No RESULT_FILE key",
        ),
        pytest.param(
            "RESULT_FILE:poly_%d.out",
            None,
            id="This should not fail",
        ),
    ],
)
def test_malformed_or_missing_gen_data_result_file(result_file, error_message):
    config_line = f"RES {result_file} REPORT_STEPS:0 INPUT_FORMAT:ASCII"
    if error_message:
        with pytest.raises(
            ConfigValidationError,
            match=error_message,
        ):
            GenDataConfig.from_config_dict({"GEN_DATA": [config_line.split()]})
    else:
        GenDataConfig.from_config_dict({"GEN_DATA": [config_line.split()]})
