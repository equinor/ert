import pytest

from ert.config.parsing import ConfigValidationError
from ert.config.parsing._parse_zonemap import parse_zonemap


def test_zone_map_line_must_start_with_an_integer():
    with pytest.raises(ConfigValidationError, match="must be an integer"):
        parse_zonemap("", "not_an_integer zone1 zone2")


def test_that_a_zonemap_line_must_contain_at_least_one_zone():
    with pytest.raises(ConfigValidationError, match="Number of zonenames"):
        parse_zonemap("", "1 \n")


def test_that_a_zonemap_maps_layer_numbers_to_zone_names():
    assert parse_zonemap("", "1 zone1 zone2\n2 zone2\n") == {
        1: ["zone1", "zone2"],
        2: ["zone2"],
    }


def test_that_whitespace_lines_are_ignored():
    assert parse_zonemap("", "1 zone1 zone2\n  \n2 zone2\n") == {
        1: ["zone1", "zone2"],
        2: ["zone2"],
    }


def test_layer_numbers_are_one_indexed():
    with pytest.raises(ConfigValidationError, match="at least 1"):
        parse_zonemap("", "0 zone1 zone2")
    with pytest.raises(ConfigValidationError, match="at least 1"):
        parse_zonemap("", "-1 zone1 zone2")
