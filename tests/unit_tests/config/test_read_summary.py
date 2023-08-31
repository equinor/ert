from datetime import datetime, timedelta
from itertools import zip_longest

import hypothesis.strategies as st
import pytest
from hypothesis import given
from resdata.summary import Summary, SummaryVarType
from resfo import Format

from ert.config._read_summary import _SummaryType, make_summary_key, read_summary

from .summary_generator import (
    inter_region_summary_variables,
    summaries,
    summary_variables,
)


def to_ecl(st: _SummaryType) -> SummaryVarType:
    if st == _SummaryType.AQUIFER:
        return SummaryVarType.RD_SMSPEC_AQUIFER_VAR
    if st == _SummaryType.BLOCK:
        return SummaryVarType.RD_SMSPEC_BLOCK_VAR
    if st == _SummaryType.COMPLETION:
        return SummaryVarType.RD_SMSPEC_COMPLETION_VAR
    if st == _SummaryType.FIELD:
        return SummaryVarType.RD_SMSPEC_FIELD_VAR
    if st == _SummaryType.GROUP:
        return SummaryVarType.RD_SMSPEC_GROUP_VAR
    if st == _SummaryType.LOCAL_BLOCK:
        return SummaryVarType.RD_SMSPEC_LOCAL_BLOCK_VAR
    if st == _SummaryType.LOCAL_COMPLETION:
        return SummaryVarType.RD_SMSPEC_LOCAL_COMPLETION_VAR
    if st == _SummaryType.LOCAL_WELL:
        return SummaryVarType.RD_SMSPEC_LOCAL_WELL_VAR
    if st == _SummaryType.NETWORK:
        return SummaryVarType.RD_SMSPEC_NETWORK_VAR
    if st == _SummaryType.SEGMENT:
        return SummaryVarType.RD_SMSPEC_SEGMENT_VAR
    if st == _SummaryType.WELL:
        return SummaryVarType.RD_SMSPEC_WELL_VAR
    if st == _SummaryType.REGION:
        return SummaryVarType.RD_SMSPEC_REGION_VAR
    if st == _SummaryType.INTER_REGION:
        return SummaryVarType.RD_SMSPEC_REGION_2_REGION_VAR
    if st == _SummaryType.OTHER:
        return SummaryVarType.RD_SMSPEC_MISC_VAR


@pytest.mark.parametrize("keyword", ["AAQR", "AAQT"])
def test_aquifer_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_AQUIFER_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.AQUIFER


@pytest.mark.parametrize("keyword", ["BOSAT"])
def test_block_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_BLOCK_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.BLOCK


@pytest.mark.parametrize("keyword", ["LBOSAT"])
def test_local_block_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_LOCAL_BLOCK_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.LOCAL_BLOCK


@pytest.mark.parametrize("keyword", ["CGORL"])
def test_completion_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_COMPLETION_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.COMPLETION


@pytest.mark.parametrize("keyword", ["LCGORL"])
def test_local_completion_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_LOCAL_COMPLETION_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.LOCAL_COMPLETION


@pytest.mark.parametrize("keyword", ["FGOR", "FOPR"])
def test_field_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_FIELD_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.FIELD


@pytest.mark.parametrize("keyword", ["GGFT", "GOPR"])
def test_group_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_GROUP_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.GROUP


@pytest.mark.parametrize("keyword", ["NOPR", "NGPR"])
def test_network_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_NETWORK_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.NETWORK


@pytest.mark.parametrize("keyword", inter_region_summary_variables)
def test_inter_region_summary_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_REGION_2_REGION_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.INTER_REGION


@pytest.mark.parametrize("keyword", ["RORFR", "RPR", "ROPT"])
def test_region_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_REGION_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.REGION


@pytest.mark.parametrize("keyword", ["SOPR"])
def test_segment_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_SEGMENT_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.SEGMENT


@pytest.mark.parametrize("keyword", ["WOPR"])
def test_well_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_WELL_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.WELL


@pytest.mark.parametrize("keyword", ["LWOPR"])
def test_local_well_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_LOCAL_WELL_VAR
    assert _SummaryType.from_keyword(keyword) == _SummaryType.LOCAL_WELL


@given(summary_variables())
def test_that_identify_var_type_is_same_as_ecl(variable):
    assert Summary.var_type(variable) == to_ecl(_SummaryType.from_keyword(variable))


@given(st.integers(), st.text(), st.integers(), st.integers())
@pytest.mark.parametrize("keyword", ["FOPR", "NEWTON"])
def test_summary_key_format_of_field_and_misc_is_identity(
    keyword, number, name, nx, ny
):
    assert make_summary_key(keyword, number, name, nx, ny) == keyword


@given(st.integers(), st.text(), st.integers(), st.integers())
def test_network_variable_keys_has_keyword_as_summary_key(number, name, nx, ny):
    assert make_summary_key("NOPR", number, name, nx, ny) == "NOPR"


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


@given(summaries(), st.sampled_from(Format))
def test_that_reading_summaries_returns_the_contents_of_the_file(
    tmp_path_factory, summary, format
):
    tmp_path = tmp_path_factory.mktemp("summary")
    format_specifier = "F" if format == Format.FORMATTED else ""
    smspec, unsmry = summary
    unsmry.to_file(tmp_path / f"TEST.{format_specifier}UNSMRY", format)
    smspec.to_file(tmp_path / f"TEST.{format_specifier}SMSPEC", format)
    (keys, time_map, data) = read_summary(str(tmp_path / "TEST"), ["*"])

    local_name = smspec.lgrs if smspec.lgrs else []
    lis = smspec.numlx if smspec.numlx else []
    ljs = smspec.numly if smspec.numly else []
    lks = smspec.numlz if smspec.numlz else []
    keys_in_smspec = [
        x
        for x in map(
            lambda x: make_summary_key(*x[:3], smspec.nx, smspec.ny, *x[3:]),
            zip_longest(
                [k.rstrip() for k in smspec.keywords],
                smspec.region_numbers,
                smspec.well_names,
                local_name,
                lis,
                ljs,
                lks,
                fillvalue=None,
            ),
        )
    ]
    assert set(keys) == set((k for k in keys_in_smspec if k))

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
    "spec_contents, smry_contents",
    [
        (b"", b""),
        (b"1", b"1"),
        (b"\x00\x00\x00\x10", b"1"),
        (b"\x00\x00\x00\x10UNEXPECTED", b"\x00\x00\x00\x10UNEXPECTED"),
        (
            b"\x00\x00\x00\x10UNEXPECTED",
            b"\x00\x00\x00\x10KEYWORD1" + (2200).to_bytes(4, byteorder="big"),
        ),
        (
            b"\x00\x00\x00\x10FOOOOOOO\x00",
            b"\x00\x00\x00\x10FOOOOOOO"
            + (2300).to_bytes(4, byteorder="big")
            + b"INTE\x00\x00\x00\x10"
            + b"\x00" * (4 * 2300 + 4 * 6),
        ),
        (
            b"\x00\x00\x00\x10FOOOOOOO\x00\x00\x00\x01"
            + b"INTE"
            + b"\x00\x00\x00\x10"
            + (4).to_bytes(4, signed=True, byteorder="big")
            + b"\x00" * 4
            + (4).to_bytes(4, signed=True, byteorder="big"),
            b"\x00\x00\x00\x10FOOOOOOO\x00",
        ),
    ],
)
def test_that_incorrect_summary_files_raises_informative_errors(
    smry_contents, spec_contents, tmp_path
):
    (tmp_path / "test.UNSMRY").write_bytes(smry_contents)
    (tmp_path / "test.SMSPEC").write_bytes(spec_contents)

    with pytest.raises(ValueError):
        read_summary(str(tmp_path / "test"), ["*"])
