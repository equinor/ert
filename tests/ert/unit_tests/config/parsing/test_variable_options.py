import pytest

from ert.config.parsing._option_dict import parse_variable_options


@pytest.mark.parametrize(
    "input_config, expected",
    [
        (
            [["NAME", "template.txt", "kw.txt", "prior.txt"], 4],
            (["NAME", "template.txt", "kw.txt", "prior.txt"], {}),
        ),
        (
            [["NAME", "template.txt", "OPTION_1:test", "OPTION_2:prior.txt"], 4],
            (["NAME", "template.txt"], {"OPTION_1": "test", "OPTION_2": "prior.txt"}),
        ),
        (
            [["NAME", "template.txt", "OPTION_1:test", "OPTION_2:prior.txt"], 1],
            (["NAME", "template.txt"], {"OPTION_1": "test", "OPTION_2": "prior.txt"}),
        ),
        (
            [["NAME", "template.txt", "kw.txt", "prior.txt"], 5],
            (["NAME", "template.txt", "kw.txt", "prior.txt"], {}),
        ),
    ],
)
def test_parse_config(input_config, expected):
    assert parse_variable_options(*input_config) == expected


def test_that_positional_arguments_must_come_before_named_arguments():
    with pytest.raises(ValueError, match="Invalid argument 'positional'"):
        parse_variable_options(
            [
                "NAME",
                "template.txt",
                "OPTION_1:test",
                "OPTION_2:prior.txt",
                "positional",
            ],
            1,
        )
