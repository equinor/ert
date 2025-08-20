import hypothesis.strategies as st
import pytest
from hypothesis import given
from resdata.summary import Summary, SummaryVarType

from ert.config._read_summary import make_summary_key
from ert.summary_key_type import SummaryKeyType, is_rate
from tests.ert.unit_tests.config.summary_generator import (
    inter_region_summary_variables,
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
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.AQUIFER


@pytest.mark.parametrize("keyword", ["BOSAT"])
def test_block_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_BLOCK_VAR
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.BLOCK


@pytest.mark.parametrize("keyword", ["LBOSAT"])
def test_local_block_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_LOCAL_BLOCK_VAR
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.LOCAL_BLOCK


@pytest.mark.parametrize("keyword", ["CGORL"])
def test_completion_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_COMPLETION_VAR
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.COMPLETION


@pytest.mark.parametrize("keyword", ["LCGORL"])
def test_local_completion_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_LOCAL_COMPLETION_VAR
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.LOCAL_COMPLETION


@pytest.mark.parametrize("keyword", ["FGOR", "FOPR"])
def test_field_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_FIELD_VAR
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.FIELD


@pytest.mark.parametrize("keyword", ["GGFT", "GOPR"])
def test_group_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_GROUP_VAR
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.GROUP


@pytest.mark.parametrize("keyword", ["NOPR", "NGPR"])
def test_network_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_NETWORK_VAR
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.NETWORK


@pytest.mark.parametrize("keyword", inter_region_summary_variables)
def test_inter_region_summary_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_REGION_2_REGION_VAR
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.INTER_REGION


@pytest.mark.parametrize("keyword", ["RORFR", "RPR", "ROPT"])
def test_region_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_REGION_VAR
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.REGION


@pytest.mark.parametrize("keyword", ["SOPR"])
def test_segment_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_SEGMENT_VAR
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.SEGMENT


@pytest.mark.parametrize("keyword", ["WOPR"])
def test_well_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_WELL_VAR
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.WELL


@pytest.mark.parametrize("keyword", ["LWOPR"])
def test_local_well_variables_are_recognized(keyword):
    assert Summary.var_type(keyword) == SummaryVarType.RD_SMSPEC_LOCAL_WELL_VAR
    assert SummaryKeyType.from_variable(keyword) == SummaryKeyType.LOCAL_WELL


@given(summary_variables())
def test_that_identify_var_type_is_same_as_resdata(variable):
    assert Summary.var_type(variable) == to_ecl(SummaryKeyType.from_variable(variable))


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


@given(
    st.sampled_from(inter_region_summary_variables),
    st.text(),
    st.integers(),
    st.integers(),
)
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


def nonempty_string_without_whitespace():
    return st.text(
        st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P")), min_size=1
    )


@given(key=nonempty_string_without_whitespace())
def test_is_rate_does_not_raise_error(key):
    is_rate_bool = is_rate(key)
    assert isinstance(is_rate_bool, bool)


@pytest.mark.parametrize(
    "opr_summary_variable", ["FOPR", "FOPR", "GOPR", "WOPR", "COPR", "ROPR", "LCOPR"]
)
def test_that_oil_production_rate_is_a_rate(opr_summary_variable):
    assert is_rate(opr_summary_variable)


@pytest.mark.parametrize(
    "opt_summary_variable", ["FOPT", "FOPT", "GOPT", "WOPT", "COPT", "ROPT", "LCOPT"]
)
def test_that_oil_production_total_is_not_a_rate(opt_summary_variable):
    assert not is_rate(opt_summary_variable)


@pytest.mark.parametrize(
    "gpr_summary_variable", ["FGPR", "FGPR", "GGPR", "WGPR", "CGPR", "RGPR", "LCGPR"]
)
def test_that_gas_production_rate_is_a_rate(gpr_summary_variable):
    assert is_rate(gpr_summary_variable)


@pytest.mark.parametrize(
    "opt_summary_variable", ["FGPT", "FGPT", "GGPT", "WGPT", "CGPT", "RGPT", "LCGPT"]
)
def test_that_gas_production_total_is_not_a_rate(opt_summary_variable):
    assert not is_rate(opt_summary_variable)


@pytest.mark.parametrize(
    "gir_summary_variable", ["FGIR", "FGIR", "GGIR", "WGIR", "CGIR", "RGIR", "LCGIR"]
)
def test_that_gas_injection_rate_is_a_rate(gir_summary_variable):
    assert is_rate(gir_summary_variable)


@pytest.mark.parametrize(
    "git_summary_variable", ["FGIT", "FGIT", "GGIT", "WGIT", "CGIT", "RGIT", "LCGIT"]
)
def test_that_gas_injection_total_is_not_a_rate(git_summary_variable):
    assert not is_rate(git_summary_variable)


@pytest.mark.parametrize(
    "wpr_summary_variable", ["FWPR", "FWPR", "GWPR", "WWPR", "CWPR", "RWPR", "LCWPR"]
)
def test_that_water_production_rate_is_a_rate(wpr_summary_variable):
    assert is_rate(wpr_summary_variable)


@pytest.mark.parametrize(
    "wpt_summary_variable", ["FWPT", "FGPT", "GGPT", "WGPT", "CGPT", "RGPT", "LCGPT"]
)
def test_that_water_production_total_is_not_a_rate(wpt_summary_variable):
    assert not is_rate(wpt_summary_variable)


@pytest.mark.parametrize(
    "wir_summary_variable", ["FWIR", "FWIR", "GWIR", "WWIR", "CWIR", "RWIR", "LCWIR"]
)
def test_that_water_injection_rate_is_a_rate(wir_summary_variable):
    assert is_rate(wir_summary_variable)


@pytest.mark.parametrize(
    "wit_summary_variable", ["FWIT", "FWIT", "GWIT", "WWIT", "CWIT", "RWIT", "LCWIT"]
)
def test_that_water_injection_total_is_not_a_rate(wit_summary_variable):
    assert not is_rate(wit_summary_variable)


@pytest.mark.parametrize("pr_summary_variable", ["FPR", "RPR", "BPR"])
def test_that_oil_phase_pressure_is_not_a_rate(pr_summary_variable):
    assert not is_rate(pr_summary_variable)


@pytest.mark.parametrize("lpr_summary_variable", ["FLPR", "GLPR", "WLPR", "CLPR"])
def test_that_liquid_production_rate_is_a_rate(lpr_summary_variable):
    assert is_rate(lpr_summary_variable)


@pytest.mark.parametrize("lpt_summary_variable", ["FLPT", "GLPT", "WLPT", "CLPT"])
def test_that_liquid_production_total_is_not_a_rate(lpt_summary_variable):
    assert not is_rate(lpt_summary_variable)


@pytest.mark.parametrize("mwpr_summary_variable", ["FMWPR", "GMWPR"])
def test_that_number_of_production_wells_flowing_is_not_a_rate(mwpr_summary_variable):
    assert not is_rate(mwpr_summary_variable)


@pytest.mark.parametrize(
    "gor_summary_variable", ["FGOR", "GGOR", "WGOR", "CGOR", "CGORL", "SGOR"]
)
def test_that_gas_oil_ratio_is_a_rate(gor_summary_variable):
    assert is_rate(gor_summary_variable)


@pytest.mark.parametrize(
    "ogr_summary_variable", ["FOGR", "GOGR", "WOGR", "COGR", "COGRL", "SOGR"]
)
def test_that_oil_gas_ratio_is_a_rate(ogr_summary_variable):
    assert is_rate(ogr_summary_variable)


@pytest.mark.parametrize(
    "wgr_summary_variable", ["FWGR", "GWGR", "WWGR", "CWGR", "CWGRL", "SWGR"]
)
def test_that_water_gas_ratio_is_a_rate(wgr_summary_variable):
    assert is_rate(wgr_summary_variable)


@pytest.mark.parametrize(
    "glr_summary_variable", ["FGLR", "GGLR", "WGLR", "CGLR", "CGLRL"]
)
def test_that_gas_liquid_ratio_is_a_rate(glr_summary_variable):
    assert is_rate(glr_summary_variable)


@pytest.mark.parametrize(
    "wct_summary_variable", ["FWCT", "GWCT", "WWCT", "CWCT", "CWCTL", "SWCT"]
)
def test_that_water_cut_is_a_rate(wct_summary_variable):
    assert is_rate(wct_summary_variable)


@given(summary_variables())
def test_rate_determination_is_the_same_as_resdata(variable):
    assert Summary.is_rate(variable) == is_rate(variable)
