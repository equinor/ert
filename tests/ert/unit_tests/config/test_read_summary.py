from array import array
from datetime import datetime, timedelta
from itertools import zip_longest

import hypothesis.strategies as st
import pytest
import resfo
from hypothesis import given
from resfo_utilities import make_summary_key
from resfo_utilities.testing import summaries

from ert.config import InvalidResponseFile
from ert.config._read_summary import read_summary

from .summary_generator import simple_smspec, simple_unsmry


@given(summaries(), st.sampled_from(resfo.Format))
def test_that_reading_summaries_returns_the_contents_of_the_file(
    tmp_path_factory, summary, format_
):
    tmp_path = tmp_path_factory.mktemp("summary")
    format_specifier = "F" if format_ == resfo.Format.FORMATTED else ""
    smspec, unsmry = summary
    unsmry.to_file(tmp_path / f"TEST.{format_specifier}UNSMRY", format_)
    smspec.to_file(tmp_path / f"TEST.{format_specifier}SMSPEC", format_)
    (_, keys, time_map, data) = read_summary(str(tmp_path / "TEST"), ["*"])

    local_name = smspec.lgrs or []
    lis = smspec.numlx or []
    ljs = smspec.numly or []
    lks = smspec.numlz or []
    keys_in_smspec = [
        make_summary_key(*x[:3], smspec.nx, smspec.ny, *x[3:])
        for x in zip_longest(
            [k.rstrip() for k in smspec.keywords],
            smspec.region_numbers,
            [well.rstrip() for well in smspec.well_names],
            [lgr.rstrip() for lgr in local_name],
            lis,
            ljs,
            lks,
            fillvalue=None,
        )
    ]
    assert set(keys) == {k for k in keys_in_smspec if k}

    def to_date(start_date: datetime, offset: float, unit: str) -> datetime:
        if unit == "DAYS":
            return start_date + timedelta(days=offset)
        if unit == "HOURS":
            return start_date + timedelta(hours=offset)
        raise InvalidResponseFile(f"Unknown time unit {unit}")

    assert all(
        abs(actual - expected) <= timedelta(minutes=15)
        for actual, expected in zip_longest(
            time_map,
            [
                to_date(
                    smspec.start_date.to_datetime(),
                    s.ministeps[-1].params[0],
                    smspec.units[0].strip(),
                )
                for s in unsmry.steps
            ],
        )
    )
    for key, d in zip_longest(keys, data):
        index = [i for i, k in enumerate(keys_in_smspec) if k == key][-1]
        assert [s.ministeps[-1].params[index] for s in unsmry.steps] == pytest.approx(d)


@pytest.mark.parametrize(
    "spec_contents, smry_contents, error_message",
    [
        (b"", b"", "Keyword startdat missing"),
        (b"1", b"1", "Failed to read summary file"),
        (
            b"\x00\x00\x00\x10FOOOOOOO\x00\x00\x00\x01"
            + b"INTE"
            + b"\x00\x00\x00\x10"
            + (4).to_bytes(4, signed=True, byteorder="big")
            + b"\x00" * 4,
            b"",
            "Keyword startdat missing",
        ),
        (
            b"\x00\x00\x00\x10STARTDAT\x00\x00\x00\x01"
            + b"INTE"
            + b"\x00\x00\x00\x10"
            + (4).to_bytes(4, signed=True, byteorder="big")
            + b"\x00" * 4
            + (4).to_bytes(4, signed=True, byteorder="big"),
            b"",
            "contains invalid STARTDAT",
        ),
    ],
)
def test_that_incorrect_summary_files_raises_informative_errors(
    smry_contents, spec_contents, error_message, tmp_path
):
    (tmp_path / "test.UNSMRY").write_bytes(smry_contents)
    (tmp_path / "test.SMSPEC").write_bytes(spec_contents)

    with pytest.raises(InvalidResponseFile, match=error_message):
        read_summary(str(tmp_path / "test"), ["*"])


def test_truncated_summary_file_raises_invalidresponsefile(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("summary")
    simple_unsmry().to_file(tmp_path / "TEST.UNSMRY")
    simple_smspec().to_file(tmp_path / "TEST.SMSPEC")

    with open(tmp_path / "TEST.UNSMRY", "ba") as unsmry_file:
        unsmry_file.truncate(100)  # 112 bytes is the un-truncated size

    with pytest.raises(InvalidResponseFile, match="Unable to read summary data from"):
        read_summary(str(tmp_path / "TEST"), ["*"])


def test_mess_values_in_summary_files_raises_informative_errors(tmp_path):
    resfo.write(tmp_path / "test.SMSPEC", [("KEYWORDS", resfo.MESS)])
    (tmp_path / "test.UNSMRY").write_bytes(b"")

    with pytest.raises(InvalidResponseFile, match="has incorrect type MESS"):
        read_summary(str(tmp_path / "test"), ["*"])


def test_empty_keywords_in_summary_files_raises_informative_errors(tmp_path):
    resfo.write(
        tmp_path / "test.SMSPEC",
        [
            ("STARTDAT", array("i", [31, 12, 2012, 00])),
            ("KEYWORDS", ["TIME    ", "        "]),
            ("UNITS   ", ["DAYS    "]),
        ],
    )
    (tmp_path / "test.UNSMRY").write_bytes(b"")

    with pytest.warns(match="Got empty summary keyword"):
        read_summary(str(tmp_path / "test"), ["*"])


def test_missing_names_keywords_in_summary_files_raises_informative_errors(
    tmp_path,
):
    resfo.write(
        tmp_path / "test.SMSPEC",
        [
            ("STARTDAT", array("i", [31, 12, 2012, 00])),
            ("KEYWORDS", ["TIME    ", "BART    "]),
            ("UNITS   ", ["DAYS    "]),
        ],
    )
    (tmp_path / "test.UNSMRY").write_bytes(b"")

    with pytest.warns(
        match="Found block keyword without dimens in summary specification",
    ):
        read_summary(str(tmp_path / "test"), ["*"])


def test_unknown_date_unit_in_summary_files_raises_informative_errors(
    tmp_path,
):
    resfo.write(
        tmp_path / "test.SMSPEC",
        [
            ("STARTDAT", array("i", [31, 12, 2012, 00])),
            ("KEYWORDS", ["TIME    "]),
            ("UNITS   ", ["ANNUAL  "]),
        ],
    )
    (tmp_path / "test.UNSMRY").write_bytes(b"")

    with pytest.raises(
        InvalidResponseFile,
        match=r"Unknown date unit .* ANNUAL",
    ):
        read_summary(str(tmp_path / "test"), ["*"])


def test_missing_units_in_summary_files_raises_an_informative_error(
    tmp_path,
):
    resfo.write(
        tmp_path / "test.SMSPEC",
        [
            ("STARTDAT", array("i", [31, 12, 2012, 00])),
            ("KEYWORDS", ["TIME    "]),
        ],
    )
    (tmp_path / "test.UNSMRY").write_bytes(b"")

    with pytest.raises(
        InvalidResponseFile,
        match="Unit missing for TIME",
    ):
        read_summary(str(tmp_path / "test"), ["*"])


def test_missing_date_units_in_summary_files_raises_an_informative_error(
    tmp_path,
):
    resfo.write(
        tmp_path / "test.SMSPEC",
        [
            ("STARTDAT", array("i", [31, 12, 2012, 00])),
            ("KEYWORDS", ["FOPR    ", "TIME    "]),
            ("UNITS   ", ["SM3     "]),
        ],
    )
    (tmp_path / "test.UNSMRY").write_bytes(b"")

    with pytest.raises(
        InvalidResponseFile,
        match="Unit missing for TIME",
    ):
        read_summary(str(tmp_path / "test"), ["*"])


def test_missing_time_keyword_in_summary_files_raises_an_informative_error(
    tmp_path,
):
    resfo.write(
        tmp_path / "test.SMSPEC",
        [
            ("STARTDAT", array("i", [31, 12, 2012, 00])),
            ("KEYWORDS", ["FOPR    "]),
            ("UNITS   ", ["SM3     "]),
        ],
    )
    (tmp_path / "test.UNSMRY").write_bytes(b"")

    with pytest.raises(
        InvalidResponseFile,
        match="KEYWORDS did not contain TIME",
    ):
        read_summary(str(tmp_path / "test"), ["*"])


def test_missing_keywords_in_smspec_raises_informative_error(
    tmp_path,
):
    resfo.write(
        tmp_path / "test.SMSPEC",
        [
            ("STARTDAT", array("i", [31, 12, 2012, 00])),
            ("UNITS   ", ["ANNUAL  "]),
        ],
    )
    (tmp_path / "test.UNSMRY").write_bytes(b"")

    with pytest.raises(
        InvalidResponseFile,
        match="Keywords missing",
    ):
        read_summary(str(tmp_path / "test"), ["*"])
