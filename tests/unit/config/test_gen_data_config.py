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
    gdc = GenDataConfig(name=name, report_steps=report_steps)
    assert gdc.name == name
    assert gdc.report_steps == sorted(report_steps)


def test_gen_data_default_report_step():
    gen_data_default_step = GenDataConfig(name="name")
    assert not gen_data_default_step.report_steps


@pytest.mark.usefixtures("use_tmpdir")
def test_gen_data_eq_config():
    alt1 = GenDataConfig(name="ALT1", report_steps=[2, 1, 3])
    alt2 = GenDataConfig(name="ALT1", report_steps=[2, 3, 1])
    alt3 = GenDataConfig(name="ALT1", report_steps=[3])
    alt4 = GenDataConfig(name="ALT4", report_steps=[3])
    alt5 = GenDataConfig(name="ALT4", report_steps=[4])

    assert alt1 == alt2  # name and ordered steps ok
    assert alt1 != alt3  # amount steps differ
    assert alt3 != alt4  # name differ
    assert alt4 != alt5  # steps differ


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
            GenDataConfig.from_config_list(config_line.split())
    else:
        GenDataConfig.from_config_list(config_line.split())
