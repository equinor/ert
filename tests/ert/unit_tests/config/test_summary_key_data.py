import pytest
from hypothesis import given
from hypothesis import strategies as st

from ert.config.summary_key_data import (
    InvalidSummaryKeyError,
    SummaryKeyData,
    make_summary_key_data,
)

INTER_REGION_PATTERNS = [r"R.FT.*", r"R..FT.*", r"R.FR.*", r"R..FR.*", r"R.F"]


_name = st.text(min_size=1, max_size=10).filter(lambda s: ":" not in s and "-" not in s)

_other_name = _name.map(lambda s: "X" + s)

_positive_int = st.integers(min_value=0, max_value=10000)


@st.composite
def _internal_region_name(draw):
    return [
        draw(
            st.from_regex(p, fullmatch=True).map(
                lambda s: s.replace("-", "A").replace(":", "B")
            )
        )
        for p in INTER_REGION_PATTERNS
    ]


def test_that_make_summary_key_data_parses_field_keys():
    assert make_summary_key_data("FIELD") == SummaryKeyData(keyword="FIELD")


@given(_positive_int)
def test_that_make_summary_key_data_parses_region_aquifer_network_keys(number: int):
    assert make_summary_key_data(f"REGION:{number}") == SummaryKeyData(
        keyword="REGION", number=number
    )
    assert make_summary_key_data(f"AQUIFER:{number}") == SummaryKeyData(
        keyword="AQUIFER", number=number
    )
    assert make_summary_key_data(f"NETWORK:{number}") == SummaryKeyData(
        keyword="NETWORK", number=number
    )


@given(_positive_int, _positive_int, _positive_int)
def test_that_make_summary_key_data_parses_block_keys(i: int, j: int, k: int):
    assert make_summary_key_data(f"BLOCK:{i},{j},{k}") == SummaryKeyData(
        keyword="BLOCK", i=i, j=j, k=k
    )


@given(_name)
def test_that_make_summary_key_data_parses_well_keys(well: str):
    assert make_summary_key_data(f"WELL:{well}") == SummaryKeyData(
        keyword="WELL", well=well
    )


@given(_name)
def test_that_make_summary_key_data_parses_group_keys(name: str):
    assert make_summary_key_data(f"GROUP:{name}") == SummaryKeyData(
        keyword="GROUP", name=name
    )


@given(_name, _positive_int)
def test_that_make_summary_key_data_parses_segment_keys(name: str, number: int):
    assert make_summary_key_data(f"SEGMENT:{name}:{number}") == SummaryKeyData(
        keyword="SEGMENT", name=name, number=number
    )


@given(_name, _positive_int, _positive_int, _positive_int)
def test_that_make_summary_key_data_parses_completion_keys(
    name: str, i: int, j: int, k: int
):
    assert make_summary_key_data(f"COMPLETION:{name}:{i},{j},{k}") == SummaryKeyData(
        keyword="COMPLETION", name=name, i=i, j=j, k=k
    )


@given(_internal_region_name(), _positive_int, _positive_int)
def test_that_make_summary_key_data_parses_inter_region_keys(
    keywords: list[str], region1: int, region2: int
):
    for keyword in keywords:
        assert make_summary_key_data(
            f"{keyword}:{region1}-{region2}"
        ) == SummaryKeyData(keyword=keyword, region1=region1, region2=region2)


@given(_name, _name)
def test_that_make_summary_key_data_parses_local_well_keys(lgr_name: str, name: str):
    assert make_summary_key_data(f"LW:{lgr_name}:{name}") == SummaryKeyData(
        keyword="LW", lgr_name=lgr_name, name=name
    )


@given(_name, _positive_int, _positive_int, _positive_int)
def test_that_make_summary_key_data_parses_local_block_keys(
    lgr_name: str, i: int, j: int, k: int
):
    assert make_summary_key_data(f"LB:{lgr_name}:{i},{j},{k}") == SummaryKeyData(
        keyword="LB", lgr_name=lgr_name, i=i, j=j, k=k
    )


@given(_name, _name, _positive_int, _positive_int, _positive_int)
def test_that_make_summary_key_data_parses_local_completion_keys(
    lgr_name: str, name: str, i: int, j: int, k: int
):
    assert make_summary_key_data(f"LC:{lgr_name}:{name}:{i},{j},{k}") == SummaryKeyData(
        keyword="LC", lgr_name=lgr_name, name=name, i=i, j=j, k=k
    )


@given(_other_name, _name)
def test_that_make_summary_key_data_parses_other_keys(other: str, name: str):
    assert make_summary_key_data(f"{other}:{name}") == SummaryKeyData(keyword=other)


def test_that_make_summary_key_data_raises_for_empty_keys():
    with pytest.raises(InvalidSummaryKeyError, match="Invalid summary key "):
        make_summary_key_data("")
