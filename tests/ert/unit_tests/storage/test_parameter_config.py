import pytest

from ert.config.parameter_config import parse_config


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
    assert parse_config(*input_config) == expected


def test_positional_after_named():
    with pytest.raises(ValueError, match="Invalid argument 'positional'"):
        parse_config(
            [
                "NAME",
                "template.txt",
                "OPTION_1:test",
                "OPTION_2:prior.txt",
                "positional",
            ],
            1,
        )
