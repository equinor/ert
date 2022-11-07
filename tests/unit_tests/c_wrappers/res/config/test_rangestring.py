import pytest

from ert._c_wrappers.config.rangestring import (
    mask_to_rangestring,
    rangestring_to_list,
    rangestring_to_mask,
)


def testMaskToRangeConversion():
    cases = (
        ([0, 1, 0, 1, 1, 1, 0], "1, 3-5"),
        ([0, 1, 0, 1, 1, 1, 1], "1, 3-6"),
        ([0, 1, 0, 1, 0, 1, 0], "1, 3, 5"),
        ([1, 1, 0, 0, 1, 1, 1, 0, 1], "0-1, 4-6, 8"),
        ([1, 1, 1, 1, 1, 1, 1], "0-6"),
        ([0, 0, 0, 0, 0, 0, 0], ""),
        ([True, False, True, True], "0, 2-3"),
        ([], ""),
    )

    def nospaces(s):
        return "".join(s.split())

    for mask, expected in cases:
        rngstr = mask_to_rangestring(mask)
        assert nospaces(rngstr) == nospaces(expected), (
            f"Mask {mask} was converted to {rngstr} which is different "
            f"from the expected range {expected}"
        )


@pytest.mark.parametrize(
    "mask, expected_string",
    [
        ([], ""),
        ([True], "0"),
        ([1], "0"),
        ([2], "0"),  # Any non-null means True
        ([-1], "0"),
        ([-0], ""),
        ([0], ""),
        ([0, 0], ""),
        ([1, 1], "0-1"),
        ([0, 1], "1"),
        ([1, 0], "0"),
        ([1, 1, 0, 0, 1, 1], "0-1, 4-5"),
    ],
)
def test_mask_to_rangestring(mask, expected_string):
    assert mask_to_rangestring(mask) == expected_string
    assert mask_to_rangestring([bool(value) for value in mask]) == expected_string


@pytest.mark.parametrize(
    "rangestring, length, expected_mask",
    [
        ("", 0, []),
        ("", 1, [False]),
        ("", 2, [False, False]),
        ("0", 1, [True]),
        ("0-0", 1, [True]),
        ("0-1", 2, [True, True]),
        ("0 - 1", 2, [True, True]),
        ("0,1", 2, [True, True]),
        ("0,1-1", 2, [True, True]),
        ("  0 ,   1 ", 2, [True, True]),
        ("1", 2, [False, True]),
        ("0,0-1", 2, [True, True]),  # overlaps allowed
    ],
)
def test_rangestring_to_mask(rangestring, length, expected_mask):
    assert rangestring_to_mask(rangestring, length) == expected_mask


@pytest.mark.parametrize(
    "rangestring, length",
    [
        ("a", 0),
        ("*", 0),
        ("-", 0),
        ("0-", 0),
        ("-1", 0),
        ("0-1", 1),
        ("0-1-1", 1),
        ("0--1", 1),
        ("1-0", 1),
        ("0-2", 1),
    ],
)
def test_rangestring_to_mask_errors(rangestring, length):
    with pytest.raises(ValueError):
        rangestring_to_mask(rangestring, length)


@pytest.mark.parametrize(
    "rangestring, expected",
    [
        ("0-1", [0, 1]),
        ("0-3", [0, 1, 2, 3]),
        ("0-1, 2, 3", [0, 1, 2, 3]),
        ("0-1,3-5", [0, 1, 3, 4, 5]),
        ("0-1, 4, 3-5", [0, 1, 3, 4, 5]),
        ("1,2,3", [1, 2, 3]),
        ("1, 3-5 ,2", [1, 2, 3, 4, 5]),
        ("1, 3-5 ,2", [1, 2, 3, 4, 5]),
    ],
)
def test_rangestring_to_list(rangestring, expected):
    assert rangestring_to_list(rangestring) == expected


@pytest.mark.parametrize(
    "rangestring",
    ["a", "*", "-", "0-", "-1", "0-a", "0-1-2", "0--1", "1-0", "0-2.2"],
)
def test_rangestring_to_list_errors(rangestring):
    with pytest.raises(ValueError):
        rangestring_to_list(rangestring)
