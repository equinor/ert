import pytest

from ert.validation import ActiveRange


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
    assert ActiveRange.validate_rangestring(rangestring) == rangestring
    assert ActiveRange.validate_rangestring_vs_length(rangestring, length) == (
        rangestring,
        length,
    )


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
        # At least one of these two must fail for the test dataset:
        ActiveRange.validate_rangestring(rangestring)
        ActiveRange.validate_rangestring_vs_length(rangestring, length)


@pytest.mark.parametrize(
    "mask, rangestring, length, expected_mask, expected_rangestring",
    [
        ([True, False], None, None, [True, False], "0"),
        ([True, False], None, 2, [True, False], "0"),
        (None, "0-1", 3, [True, True, False], "0-1"),
        (None, "0-1,1", 3, [True, True, False], "0-1"),
    ],
)
def test_activerange(mask, rangestring, length, expected_mask, expected_rangestring):
    activerange = ActiveRange(mask=mask, rangestring=rangestring, length=length)
    assert activerange.mask == expected_mask
    assert activerange.rangestring == expected_rangestring
    assert len(activerange) == len(expected_mask)


@pytest.mark.parametrize(
    "mask, rangestring, length",
    [
        (None, None, None),
        (None, "0-1", None),
        (None, None, 3),
        ([True, True], None, 1),
        ([True, True], "0", None),
        ([True, True], "0", 2),
    ],
)
def test_activerange_initerrors(mask, rangestring, length):
    with pytest.raises(ValueError):
        ActiveRange(mask=mask, rangestring=rangestring, length=length)
