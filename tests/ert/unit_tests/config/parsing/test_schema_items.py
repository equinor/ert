import pytest

from ert.config.parsing import ConfigValidationError, parse_contents
from ert.config.parsing.config_schema import ConfigSchemaDict
from ert.config.parsing.config_schema_item import SchemaItem


def test_that_schema_argc_min_one_raises_on_zero_arguments():
    schema = ConfigSchemaDict(OPTIONS=SchemaItem(kw="OPTIONS", argc_min=1))
    with pytest.raises(ConfigValidationError, match="must have at least 1 argument"):
        parse_contents("OPTIONS", schema, "dummy_filename")


def test_that_a_schema_item_can_contain_only_options():
    all_options_item = SchemaItem(
        kw="OPTIONS",
        required_set=False,
        multi_occurrence=True,
        options_after=0,
        argc_min=1,
        argc_max=1,
    )
    schema = ConfigSchemaDict(OPTIONS=all_options_item)

    parsed = parse_contents(
        """
            OPTIONS OPT1:VAL1 OPT2:VAL2
            OPTIONS OPT1:VAL3 OPT2:VAL4
            """,
        schema,
        "unused",
    )

    del parsed["DEFINE"]

    assert parsed == {
        "OPTIONS": [{"OPT1": "VAL1", "OPT2": "VAL2"}, {"OPT1": "VAL3", "OPT2": "VAL4"}],
    }
