from datetime import datetime, timedelta
from itertools import zip_longest

import hypothesis.strategies as st
import pytest
import resfo
from hypothesis import given
from resdata.summary import Summary, SummaryVarType

from ert.config._read_summary import make_summary_key, read_summary
from ert.summary_key_type import SummaryKeyType

from .summary_generator import (
    inter_region_summary_variables,
    summaries,
    summary_variables,
)


def to_ecl(st: SummaryKeyType) -> SummaryVarType:
    if st == SummaryKeyType.AQUIFER:
        return SummaryVarType.RD_SMSPEC_AQUIFER_VAR
    if st == SummaryKeyType.BLOCK:
        return SummaryVarType.RD_SMSPEC_BLOCK_VAR
    if st == SummaryKeyType.COMPLETION:
        return SummaryVarType.RD_SMSPEC_COMPLETION_VAR
    if st == SummaryKeyType.FIELD:
        return SummaryVarType.RD_SMSPEC_FIELD_VAR
    if st == SummaryKeyType.GROUP:
        return SummaryVarType.RD_SMSPEC_GROUP_VAR
    if st == SummaryKeyType.LOCAL_BLOCK:
        return SummaryVarType.RD_SMSPEC_LOCAL_BLOCK_VAR
    if st == SummaryKeyType.LOCAL_COMPLETION:
        return SummaryVarType.RD_SMSPEC_LOCAL_COMPLETION_VAR
    if st == SummaryKeyType.LOCAL_WELL:
        return SummaryVarType.RD_SMSPEC_LOCAL_WELL_VAR
    if st == SummaryKeyType.NETWORK:
        return SummaryVarType.RD_SMSPEC_NETWORK_VAR
    if st == SummaryKeyType.SEGMENT:
        return SummaryVarType.RD_SMSPEC_SEGMENT_VAR
    if st == SummaryKeyType.WELL:
        return SummaryVarType.RD_SMSPEC_WELL_VAR
    if st == SummaryKeyType.REGION:
        return SummaryVarType.RD_SMSPEC_REGION_VAR
    if st == SummaryKeyType.INTER_REGION:
        return SummaryVarType.RD_SMSPEC_REGION_2_REGION_VAR
    if st == SummaryKeyType.OTHER:
        return SummaryVarType.RD_SMSPEC_MISC_VAR


@pytest.mark.parametrize("keyword", ["AAQR", "AAQT"])
def test_aquifer_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_AQUIFER_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.AQUIFER


@pytest.mark.parametrize("keyword", ["BOSAT"])
def test_block_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_BLOCK_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.BLOCK


@pytest.mark.parametrize("keyword", ["LBOSAT"])
def test_local_block_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_LOCAL_BLOCK_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.LOCAL_BLOCK


@pytest.mark.parametrize("keyword", ["CGORL"])
def test_completion_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_COMPLETION_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.COMPLETION


@pytest.mark.parametrize("keyword", ["LCGORL"])
def test_local_completion_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_LOCAL_COMPLETION_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.LOCAL_COMPLETION


@pytest.mark.parametrize("keyword", ["FGOR", "FOPR"])
def test_field_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_FIELD_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.FIELD


@pytest.mark.parametrize("keyword", ["GGFT", "GOPR"])
def test_group_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_GROUP_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.GROUP


@pytest.mark.parametrize("keyword", ["NOPR", "NGPR"])
def test_network_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_NETWORK_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.NETWORK


@pytest.mark.parametrize("keyword", inter_region_summary_variables)
def test_inter_region_summary_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_REGION_2_REGION_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.INTER_REGION


@pytest.mark.parametrize("keyword", ["RORFR", "RPR", "ROPT"])
def test_region_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_REGION_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.REGION


@pytest.mark.parametrize("keyword", ["SOPR"])
def test_segment_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_SEGMENT_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.SEGMENT


@pytest.mark.parametrize("keyword", ["WOPR"])
def test_well_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_WELL_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.WELL


@pytest.mark.parametrize("keyword", ["LWOPR"])
def test_local_well_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_LOCAL_WELL_VAR
    assert SummaryKeyType.from_keyword(keyword) == SummaryKeyType.LOCAL_WELL


@given(summary_variables())
def test_that_identify_var_type_is_same_as_ecl(variable):
    assert Summary.var_type(variable) == to_ecl(SummaryKeyType.from_keyword(variable))


@given(st.integers(), st.text(), st.integers(), st.integers())
@pytest.mark.parametrize("keyword", ["FOPR", "NEWTON"])
def test_summary_key_format_of_field_and_misc_is_identity(
    keyword, number, name, nx, ny
):
    assert make_summary_key(keyword, number, name, nx, ny) == keyword


@given(st.integers(), st.text(), st.integers(), st.integers())
def test_network_variable_keys_has_keyword_as_summary_key(number, name, nx, ny):
    assert make_summary_key("NOPR", number, name, nx, ny) == f"NOPR:{name}"


@given(st.integers(), st.text(), st.integers(), st.integers())
@pytest.mark.parametrize("keyword", ["GOPR", "WOPR"])
def test_group_and_well_have_named_format(keyword, number, name, nx, ny):
    assert make_summary_key(keyword, number, name, nx, ny) == f"{keyword}:{name}"


@given(st.text(), st.integers(), st.integers())
@pytest.mark.parametrize("keyword", inter_region_summary_variables)
def test_inter_region_summary_format_contains_in_and_out_regions(keyword, name, nx, ny):
    number = 3014660
    assert make_summary_key(keyword, number, name, nx, ny) == f"{keyword}:4-82"


@given(name=st.text())
@pytest.mark.parametrize("keyword", ["BOPR", "BOSAT"])
@pytest.mark.parametrize(
    "nx,ny,number,indices",
    [
        (1, 1, 1, "1,1,1"),
        (2, 1, 2, "2,1,1"),
        (1, 2, 2, "1,2,1"),
        (3, 2, 3, "3,1,1"),
        (3, 2, 9, "3,1,2"),
    ],
)
def test_block_summary_format_have_cell_index(keyword, number, indices, name, nx, ny):
    assert make_summary_key(keyword, number, name, nx, ny) == f"{keyword}:{indices}"


@given(name=st.text())
@pytest.mark.parametrize("keyword", ["COPR"])
@pytest.mark.parametrize(
    "nx,ny,number,indices",
    [
        (1, 1, 1, "1,1,1"),
        (2, 1, 2, "2,1,1"),
        (1, 2, 2, "1,2,1"),
        (3, 2, 3, "3,1,1"),
        (3, 2, 9, "3,1,2"),
    ],
)
def test_completion_summary_format_have_cell_index_and_name(
    keyword, number, indices, name, nx, ny
):
    assert (
        make_summary_key(keyword, number, name, nx, ny) == f"{keyword}:{name}:{indices}"
    )


@pytest.mark.parametrize("keyword", ["LBWPR"])
@pytest.mark.parametrize(
    "li,lj,lk,lgr_name,indices",
    [
        (1, 1, 1, "LGRNAME", "1,1,1"),
        (2, 1, 1, "LGRNAME", "2,1,1"),
        (1, 2, 1, "LGRNAME", "1,2,1"),
        (3, 1, 1, "LGRNAME", "3,1,1"),
        (3, 1, 2, "LGRNAME", "3,1,2"),
    ],
)
def test_local_block_summary_format_have_cell_index_and_name(
    keyword, lgr_name, indices, li, lj, lk
):
    assert (
        make_summary_key(keyword, li=li, lj=lj, lk=lk, lgr_name=lgr_name)
        == f"{keyword}:{lgr_name}:{indices}"
    )


@given(name=st.text(), lgr_name=st.text())
@pytest.mark.parametrize("keyword", ["LCOPR"])
@pytest.mark.parametrize(
    "li,lj,lk,indices",
    [
        (1, 1, 1, "1,1,1"),
        (2, 1, 1, "2,1,1"),
        (1, 2, 1, "1,2,1"),
        (3, 1, 1, "3,1,1"),
        (3, 1, 2, "3,1,2"),
    ],
)
def test_local_completion_summary_format_have_cell_index_and_name(
    keyword, name, lgr_name, indices, li, lj, lk
):
    assert (
        make_summary_key(keyword, name=name, li=li, lj=lj, lk=lk, lgr_name=lgr_name)
        == f"{keyword}:{lgr_name}:{name}:{indices}"
    )


@given(name=st.text(), lgr_name=st.text())
@pytest.mark.parametrize("keyword", ["LWWPR"])
def test_local_well_summary_format_have_cell_index_and_name(keyword, name, lgr_name):
    assert (
        make_summary_key(keyword, name=name, lgr_name=lgr_name)
        == f"{keyword}:{lgr_name}:{name}"
    )


@pytest.mark.integration_test
@given(summaries(), st.sampled_from(resfo.Format))
def test_that_reading_summaries_returns_the_contents_of_the_file(
    tmp_path_factory, summary, format
):
    tmp_path = tmp_path_factory.mktemp("summary")
    format_specifier = "F" if format == resfo.Format.FORMATTED else ""
    smspec, unsmry = summary
    unsmry.to_file(tmp_path / f"TEST.{format_specifier}UNSMRY", format)
    smspec.to_file(tmp_path / f"TEST.{format_specifier}SMSPEC", format)
    (_, keys, time_map, data) = read_summary(str(tmp_path / "TEST"), ["*"])

    local_name = smspec.lgrs if smspec.lgrs else []
    lis = smspec.numlx if smspec.numlx else []
    ljs = smspec.numly if smspec.numly else []
    lks = smspec.numlz if smspec.numlz else []
    keys_in_smspec = [
        make_summary_key(*x[:3], smspec.nx, smspec.ny, *x[3:])
        for x in zip_longest(
            [k.rstrip() for k in smspec.keywords],
            smspec.region_numbers,
            [w.rstrip() for w in smspec.well_names],
            [l.rstrip() for l in local_name],
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
        raise ValueError(f"Unknown time unit {unit}")

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

    with pytest.raises(ValueError, match=error_message):
        read_summary(str(tmp_path / "test"), ["*"])


def test_mess_values_in_summary_files_raises_informative_errors(tmp_path):
    resfo.write(tmp_path / "test.SMSPEC", [("KEYWORDS", resfo.MESS)])
    (tmp_path / "test.UNSMRY").write_bytes(b"")

    with pytest.raises(ValueError, match="has incorrect type MESS"):
        read_summary(str(tmp_path / "test"), ["*"])


def test_empty_keywords_in_summary_files_raises_informative_errors(tmp_path):
    resfo.write(
        tmp_path / "test.SMSPEC",
        [("STARTDAT", array("i", [31, 12, 2012, 00])), ("KEYWORDS", ["        "])],
    )
    (tmp_path / "test.UNSMRY").write_bytes(b"")

    with pytest.raises(ValueError, match="Got empty summary keyword"):
        read_summary(str(tmp_path / "test"), ["*"])


from array import array


def test_missing_names_keywords_in_summary_files_raises_informative_errors(
    tmp_path,
):
    resfo.write(
        tmp_path / "test.SMSPEC",
        [("STARTDAT", array("i", [31, 12, 2012, 00])), ("KEYWORDS", ["BART    "])],
    )
    (tmp_path / "test.UNSMRY").write_bytes(b"")

    with pytest.raises(
        ValueError,
        match="Found block keyword in summary specification without dimens keyword",
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
        ValueError,
        match="Unknown date unit .* ANNUAL",
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
        ValueError,
        match="Keyword units",
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
        ValueError,
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
        ValueError,
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
        ValueError,
        match="Keywords missing",
    ):
        read_summary(str(tmp_path / "test"), ["*"])


def test_that_ambiguous_case_restart_raises_an_informative_error(
    tmp_path,
):
    (tmp_path / "test.UNSMRY").write_bytes(b"")
    (tmp_path / "test.FUNSMRY").write_bytes(b"")
    (tmp_path / "test.smspec").write_bytes(b"")
    (tmp_path / "test.Smspec").write_bytes(b"")

    with pytest.raises(
        ValueError,
        match="Ambiguous reference to unified summary",
    ):
        read_summary(str(tmp_path / "test"), ["*"])
