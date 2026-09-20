import pytest

from ert.gui.utils import (
    LONGEST_DEFAULT_EXPERIMENT_NAME,
    truncate_dropdown_item,
    truncate_experiment_name,
    truncate_string,
)


@pytest.mark.parametrize(
    ("input_string", "max_length", "expected_output"),
    [
        ("short", 10, "short"),
        ("exactlyten", 10, "exactlyten"),
        (
            "exactlyis11",
            10,
            "exa...is11",
        ),
    ],
)
def test_that_truncate_string_truncates_when_appropriate(
    input_string, max_length, expected_output
):
    truncatedString = truncate_string(input_string, max_length)
    assert truncatedString == expected_output
    if input_string != expected_output:
        assert len(truncatedString) == max_length
    else:
        assert len(truncatedString) == len(input_string)


@pytest.mark.parametrize(
    ("experiment_name", "expected_output"),
    [
        ("ensemble_information_filter", "ensemble_information_filter"),
        ("ensemble_information_filter_11", "ensemble_inf...on_filter_11"),
    ],
)
def test_that_truncate_experiment_name_to_longest_default_length(
    experiment_name, expected_output
):
    truncatedName = truncate_experiment_name(experiment_name)
    assert truncatedName == expected_output
    if experiment_name != expected_output:
        assert len(truncatedName) == LONGEST_DEFAULT_EXPERIMENT_NAME
    else:
        assert len(truncatedName) == len(experiment_name)


@pytest.mark.parametrize(
    ("dropdown_item", "expected_output"),
    [
        ("short item", "short item"),
        ("a" * 100, "a" * 100),
        (
            "a" * 101,
            "a" * 48 + "..." + "a" * 49,
        ),
    ],
)
def test_that_truncate_dropdown_item_truncates_to_100_characters(
    dropdown_item, expected_output
):
    truncatedItem = truncate_dropdown_item(dropdown_item)
    assert truncatedItem == expected_output
    if dropdown_item != expected_output:
        assert len(truncatedItem) == 100
    else:
        assert len(truncatedItem) == len(dropdown_item)
