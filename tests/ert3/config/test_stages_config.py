import pytest
import pydantic
from ert3.config import _stages_config


def _example_config():
    return [
        {
            "name": "evaluate_polynomial",
            "environment": "polynomial",
            "input": [{"record": "coefficients", "location": "coefficients.json"}],
            "output": [{"record": "polynomial_output", "location": "output.json"}],
            "script": ["ert3.evaluator.poly:polynomial"],
        }
    ]


def example_func():
    return 123


def test_entry_point():
    config = _stages_config.load_stages_config(_example_config())
    config = config[0]
    assert config.name == "evaluate_polynomial"
    assert config.environment == "polynomial"


@pytest.mark.parametrize(
    "config, expected_error",
    (
        [{"not_a_key": "value"}, "1 validation error"],
        [[{"not_a_key": "value"}], "4 validation errors"],
    ),
)
def test_entry_point_not_valid(config, expected_error):
    with pytest.raises(pydantic.error_wrappers.ValidationError, match=expected_error):
        _stages_config.load_stages_config(config)


def test_step_valid():
    config = _stages_config.Step(
        **{
            "name": "some_name",
            "script": ["tests.ert3.config.test_stages_config:example_func"],
            "input": [{"record": "some_record", "location": "some_location"}],
            "output": [{"record": "some_record", "location": "some_location"}],
        }
    )
    assert config.name == "some_name"
    assert config.script[0]() == example_func()


@pytest.mark.parametrize(
    "script, expected_error",
    [
        (["not.a.module:some_func"], "No module named: not.a.module"),
        (
            ["tests.ert3.config.test_stages_config:some_func"],
            "No function named: some_func",
        ),
        (["not.a.module"], "must be: some.module:function_name"),
    ],
)
def test_step_invalid_script(script, expected_error):
    config = {
        "name": "some_name",
        "input": [{"record": "some_record", "location": "some_location"}],
        "output": [{"record": "some_record", "location": "some_location"}],
    }
    config.update({"script": script})
    with pytest.raises(pydantic.error_wrappers.ValidationError, match=expected_error):
        _stages_config.Step(**config)


def test_stages_config():
    _stages_config.StagesConfig.parse_obj(
        [
            {
                "name": "some_name",
                "script": ["tests.ert3.config.test_stages_config:example_func"],
                "input": [{"record": "some_record", "location": "some_location"}],
                "output": [{"record": "some_record", "location": "some_location"}],
            }
        ]
    )


def test_stages_get_script():
    config = _stages_config.StagesConfig.parse_obj(
        [
            {
                "name": "some_name",
                "script": ["tests.ert3.config.test_stages_config:example_func"],
                "input": [{"record": "some_record", "location": "some_file"}],
                "output": [{"record": "some_record", "location": "some_file"}],
            }
        ]
    )
    step = config.step_from_key("some_name")
    assert step.script[0]() == example_func()
