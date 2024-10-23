import os
from contextlib import suppress
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given

from ert.config import (
    ConfigValidationError,
    ErtConfig,
    InvalidResponseFile,
    SummaryConfig,
)

from .summary_generator import summaries


def test_bad_user_config_file_error_message(tmp_path):
    (tmp_path / "test.ert").write_text("NUM_REALIZATIONS 10\n SUMMARY FOPR")
    with pytest.raises(
        ConfigValidationError,
        match=r"Line 2 .* When using SUMMARY keyword,"
        " the config must also specify ECLBASE",
    ):
        _ = ErtConfig.from_file(str(tmp_path / "test.ert"))


@given(summaries(summary_keys=st.just(["WOPR:OP1"])))
@pytest.mark.usefixtures("use_tmpdir")
def test_rading_empty_summaries_raises(wopr_summary):
    smspec, unsmry = wopr_summary
    smspec.to_file("CASE.SMSPEC")
    unsmry.to_file("CASE.UNSMRY")
    with pytest.raises(InvalidResponseFile, match="Did not find any summary values"):
        SummaryConfig("summary", "CASE", ["WWCT:OP1"], None).read_from_file(".", 0)


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
        SummaryConfig("summary", ["CASE"], ["FOPR"]).read_from_file(os.getcwd(), 1)


def test_that_read_file_does_not_raise_unexpected_exceptions_on_missing_file(tmpdir):
    with pytest.raises(FileNotFoundError):
        SummaryConfig("summary", ["NOT_CASE"], ["FOPR"]).read_from_file(tmpdir, 1)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_read_file_does_not_raise_unexpected_exceptions_on_missing_directory(
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        SummaryConfig("summary", ["CASE"], ["FOPR"]).read_from_file(
            str(tmp_path / "DOES_NOT_EXIST"), 1
        )
