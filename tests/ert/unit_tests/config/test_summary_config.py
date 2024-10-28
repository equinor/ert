import os
from contextlib import suppress
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from ert.config import (
    ConfigValidationError,
    ErtConfig,
    InvalidResponseFile,
    SummaryConfig,
)

from .summary_generator import summaries


def test_bad_user_config_file_error_message():
    with pytest.raises(
        ConfigValidationError,
        match=r"Line 2 .* When using SUMMARY keyword,"
        " the config must also specify ECLBASE",
    ):
        _ = ErtConfig.from_file_contents("NUM_REALIZATIONS 10\n SUMMARY FOPR")


@settings(max_examples=10)
@given(summaries(summary_keys=st.just(["WOPR:OP1"])))
@pytest.mark.usefixtures("use_tmpdir")
def test_reading_empty_summaries_raises(wopr_summary):
    smspec, unsmry = wopr_summary
    smspec.to_file("CASE.SMSPEC")
    unsmry.to_file("CASE.UNSMRY")
    with pytest.raises(InvalidResponseFile, match="Did not find any summary values"):
        SummaryConfig("summary", ["CASE"], ["WWCT:OP1"]).read_from_file(lambda x: x)


def test_summary_config_normalizes_list_of_keys():
    assert SummaryConfig("summary", "CASE", ["FOPR", "WOPR", "WOPR"]).keys == [
        "FOPR",
        "WOPR",
    ]


@pytest.mark.usefixtures("use_tmpdir")
@given(st.binary(), st.binary())
def test_that_read_file_does_not_raise_unexpected_exceptions_on_invalid_file(
    smspec, unsmry
):
    Path("CASE.UNSMRY").write_bytes(unsmry)
    Path("CASE.SMSPEC").write_bytes(smspec)
    with suppress(InvalidResponseFile):
        SummaryConfig("summary", ["CASE"], ["FOPR"]).read_from_file(
            lambda x: os.getcwd() + "/" + x
        )


def test_that_read_file_does_not_raise_unexpected_exceptions_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        SummaryConfig("summary", ["NOT_CASE"], ["FOPR"]).read_from_file(
            lambda x: str(tmp_path / x)
        )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_read_file_does_not_raise_unexpected_exceptions_on_missing_directory(
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        SummaryConfig("summary", ["CASE"], ["FOPR"]).read_from_file(
            lambda x: str(tmp_path / "DOES_NOT_EXIST" / x)
        )
