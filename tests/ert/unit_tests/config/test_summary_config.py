import os
from contextlib import suppress
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from resfo_utilities.testing import summaries

from ert.config import (
    InvalidResponseFile,
    SummaryConfig,
)


@settings(max_examples=10)
@given(summaries(summary_keys=st.just(["WOPR:OP1"])))
@pytest.mark.usefixtures("use_tmpdir")
def test_reading_empty_summaries_raises(wopr_summary):
    smspec, unsmry = wopr_summary
    smspec.to_file("CASE.SMSPEC")
    unsmry.to_file("CASE.UNSMRY")
    with pytest.raises(InvalidResponseFile, match="Did not find any summary values"):
        SummaryConfig(
            name="summary", input_files=["CASE"], keys=["WWCT:OP1"]
        ).read_from_file(".", 0, 0)


def test_summary_config_normalizes_list_of_keys():
    assert SummaryConfig(
        name="summary", input_files=["CASE"], keys=["FOPR", "WOPR", "WOPR"]
    ).keys == [
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
        SummaryConfig(
            name="summary", input_files=["CASE"], keys=["FOPR"]
        ).read_from_file(os.getcwd(), 1, 0)


def test_that_read_file_does_not_raise_unexpected_exceptions_on_missing_file(tmpdir):
    with pytest.raises(FileNotFoundError):
        SummaryConfig(
            name="summary", input_files=["NOT_CASE"], keys=["FOPR"]
        ).read_from_file(tmpdir, 1, 0)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_read_file_does_not_raise_unexpected_exceptions_on_missing_directory(
    tmp_path,
):
    with pytest.raises(FileNotFoundError):
        SummaryConfig(
            name="summary", input_files=["CASE"], keys=["FOPR"]
        ).read_from_file(str(tmp_path / "DOES_NOT_EXIST"), 1, 0)
