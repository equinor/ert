import os
import re
import pathlib
import stat
from typing import Any, Callable, Dict, List, Tuple, Type
from unittest.mock import patch

import pytest

import ert3
import ert


def create_mock_script(path, data=""):
    script_file = pathlib.Path(path)
    script_file.write_text(data)
    st = os.stat(script_file)
    os.chmod(script_file, st.st_mode | stat.S_IEXEC)


@pytest.fixture()
def base_unix_stage_config(tmpdir):
    with tmpdir.as_cwd():
        config = [
            {
                "name": "unix_stage",
                "input": [
                    {
                        "name": "some_record",
                        "transformation": {
                            "location": "some_location",
                            "type": "serialization",
                        },
                    }
                ],
                "output": [
                    {
                        "name": "some_record",
                        "transformation": {
                            "location": "some_location",
                            "type": "serialization",
                        },
                    }
                ],
                "transportable_commands": [{"name": "poly", "location": "poly.py"}],
                "script": ["poly --help"],
            }
        ]
        create_mock_script("poly.py")
        yield config


@pytest.fixture()
def base_function_stage_config(tmpdir):
    config = [
        {
            "name": "function_stage",
            "input": [{"name": "some_record"}],
            "output": [
                {
                    "name": "some_record",
                }
            ],
            "function": "builtins:sum",
        }
    ]
    yield config


@pytest.fixture(params=["base_unix_stage_config", "base_function_stage_config"])
def base_stage_config(request):
    yield request.getfixturevalue(request.param)


def test_entry_point(base_unix_stage_config, plugin_registry):
    config = ert3.config.load_stages_config(
        base_unix_stage_config, plugin_registry=plugin_registry
    )
    config = config[0]
    assert config.name == "unix_stage"
    assert config.script[0] == "poly --help"


@pytest.mark.parametrize(
    "config, expected_error",
    (
        [{"not_a_key": "value"}, "1 validation error"],
        [[{"not_a_key": "value"}], "11 validation errors"],
    ),
)
def test_entry_point_not_valid(config, expected_error, plugin_registry):
    with pytest.raises(ert.exceptions.ConfigValidationError, match=expected_error):
        ert3.config.load_stages_config(config, plugin_registry=plugin_registry)


def test_check_loaded_mime_types_new_default(base_unix_stage_config, plugin_registry):
    """If an IO's default mime is set to something other than an empty string,
    the `_ensure_mime` validator will not modify the mime,
    which may lead to an invalid mime.
    """

    ert3.config.create_stages_config(plugin_registry=plugin_registry)
    from ert3.config._config_plugin_registry import (
        FullSerializationTransformationConfig,
    )

    raw_config = base_unix_stage_config

    with patch.object(
        FullSerializationTransformationConfig.__dict__["__fields__"]["mime"],
        "default",
        "application/not_a_valid_mime",
    ):
        with pytest.raises(ert.exceptions.ConfigValidationError):
            _ = ert3.config.load_stages_config(
                raw_config, plugin_registry=plugin_registry
            )


def test_check_loaded_mime_types(base_unix_stage_config, plugin_registry):
    """Test mimetype in transportable commands, input, and output"""
    raw_config = base_unix_stage_config
    raw_config[0]["input"].append(
        {
            "name": "some_json_record",
            "transformation": {
                "location": "some_location.json",
                "type": "serialization",
            },
        }
    )
    raw_config[0]["output"].append(
        {
            "name": "some_json_record",
            "transformation": {
                "location": "some_location.json",
                "type": "serialization",
            },
        }
    )
    config = ert3.config.load_stages_config(raw_config, plugin_registry=plugin_registry)
    # Check transportable_commands
    assert (
        config[0].transportable_commands[0].mime == ert3.config.DEFAULT_RECORD_MIME_TYPE
    )
    # Check input
    assert (
        config[0].input[0].transformation.mime == ert3.config.DEFAULT_RECORD_MIME_TYPE
    )
    assert config[0].input["some_json_record"].transformation.mime == "application/json"
    # Check output
    assert (
        config[0].output[0].transformation.mime == ert3.config.DEFAULT_RECORD_MIME_TYPE
    )
    assert (
        config[0].output["some_json_record"].transformation.mime == "application/json"
    )


def test_step_multi_cmd(base_unix_stage_config, plugin_registry):
    config = base_unix_stage_config
    config[0]["transportable_commands"].append(
        {"name": "poly2", "location": "poly2", "mime": "text/x-python"}
    )

    config[0]["script"] = [
        "poly run1",
        "poly2 gogo",
        "poly run2",
        "poly2 abort",
    ]
    create_mock_script(path="poly2")

    with pytest.raises(ert.exceptions.ConfigValidationError):
        config = ert3.config.load_stages_config(config, plugin_registry=plugin_registry)


def test_step_non_existing_transportable_cmd(base_unix_stage_config, plugin_registry):
    invalid_location = "/not/a/file"
    assert not os.path.exists(invalid_location)

    config = base_unix_stage_config
    config[0]["transportable_commands"].append(
        {"name": "invalid_cmd", "location": invalid_location}
    )

    err_msg = '"/not/a/file" does not exist'
    with pytest.raises(ert.exceptions.ConfigValidationError, match=err_msg):
        ert3.config.load_stages_config(config, plugin_registry=plugin_registry)


def test_step_function_definition_error(base_function_stage_config, plugin_registry):
    config = base_function_stage_config
    config[0]["function"] = "builtinssum"
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=r"Function should be defined as module:function",
    ):
        ert3.config.load_stages_config(config, plugin_registry=plugin_registry)


def test_step_function_error(base_function_stage_config, plugin_registry):
    config = base_function_stage_config
    config[0]["function"] = "builtins:sun"
    with pytest.raises(ImportError):
        ert3.config.load_stages_config(config, plugin_registry=plugin_registry)


def test_step_unix_and_function(base_unix_stage_config, plugin_registry):
    config = base_unix_stage_config
    config[0].update({"function": "builtins:sum"})
    with pytest.raises(
        ert.exceptions.ConfigValidationError, match=r"extra fields not permitted"
    ):
        ert3.config.load_stages_config(config, plugin_registry=plugin_registry)


def test_step_function_module_error(base_function_stage_config, plugin_registry):
    config = base_function_stage_config
    config[0]["function"] = "builtinx:sum"
    with pytest.raises(
        ModuleNotFoundError,
        match=r"No module named 'builtinx'",
    ):
        ert3.config.load_stages_config(config, plugin_registry=plugin_registry)


def test_step_function_and_script_error(base_function_stage_config, plugin_registry):
    config = base_function_stage_config
    config[0].update({"script": ["poly --help"]})
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=r"extra fields not permitted",
    ):
        ert3.config.load_stages_config(config, plugin_registry=plugin_registry)


def test_step_function_and_command_error(base_function_stage_config, plugin_registry):
    config = base_function_stage_config
    config[0].update(
        {"transportable_commands": [{"name": "poly", "location": "poly.py"}]}
    )
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match=r"extra fields not permitted",
    ):
        ert3.config.load_stages_config(config, plugin_registry=plugin_registry)


def test_step_function_with_no_transformation(
    base_function_stage_config, plugin_registry
):
    config = base_function_stage_config
    stage_config = ert3.config.load_stages_config(
        config, plugin_registry=plugin_registry
    )

    assert stage_config[0].input[0].get_transformation_instance() == None


def test_multi_stages_get_script(
    base_unix_stage_config, base_function_stage_config, plugin_registry
):
    multi_stage_config = base_unix_stage_config + base_function_stage_config
    config = ert3.config.load_stages_config(
        multi_stage_config, plugin_registry=plugin_registry
    )

    step1 = config.step_from_key("unix_stage")
    assert [cmd for cmd in step1.script] == ["poly --help"]
    step2 = config.step_from_key("function_stage")
    assert isinstance(step2.function, Callable)
    assert step2.function.__name__ == "sum"


def test_base_immutable(base_stage_config, plugin_registry):
    config = ert3.config.load_stages_config(
        base_stage_config, plugin_registry=plugin_registry
    )

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0] = config[0]


def test_stage_immutable(base_stage_config, plugin_registry):
    config = ert3.config.load_stages_config(
        base_stage_config, plugin_registry=plugin_registry
    )

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].name = "new-name"


def test_stage_unknown_field(base_stage_config, plugin_registry):
    base_stage_config[0]["unknown"] = "field"
    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="extra fields not permitted",
    ):
        ert3.config.load_stages_config(
            base_stage_config, plugin_registry=plugin_registry
        )


def test_stage_input_immutable(base_stage_config, plugin_registry):
    config = ert3.config.load_stages_config(
        base_stage_config, plugin_registry=plugin_registry
    )
    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].input[0] = None

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].input[0].name = None


def test_stage_input_unknown_field(base_stage_config, plugin_registry):
    base_stage_config[0]["input"][0]["unknown"] = "field"

    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="extra fields not permitted",
    ):
        ert3.config.load_stages_config(
            base_stage_config, plugin_registry=plugin_registry
        )


def test_stage_output_immutable(base_stage_config, plugin_registry):
    config = ert3.config.load_stages_config(
        base_stage_config, plugin_registry=plugin_registry
    )
    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].output[0] = None

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].output[0].name = None


def test_stage_output_unknown_field(base_stage_config, plugin_registry):
    base_stage_config[0]["output"][0]["unknown"] = "field"

    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="extra fields not permitted",
    ):
        ert3.config.load_stages_config(
            base_stage_config, plugin_registry=plugin_registry
        )


def test_stage_transportable_command_immutable(base_unix_stage_config, plugin_registry):
    config = ert3.config.load_stages_config(
        base_unix_stage_config, plugin_registry=plugin_registry
    )

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].transportable_commands[0] = None

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].transportable_commands[0].name = "some-name"


def test_stage_transportable_command_unknown_field(
    base_unix_stage_config, plugin_registry
):
    base_unix_stage_config[0]["transportable_commands"][0]["unknown"] = "field"

    with pytest.raises(
        ert.exceptions.ConfigValidationError,
        match="extra fields not permitted",
    ):
        ert3.config.load_stages_config(
            base_unix_stage_config, plugin_registry=plugin_registry
        )


def test_stage_script_immutable(base_unix_stage_config, plugin_registry):
    config = ert3.config.load_stages_config(
        base_unix_stage_config, plugin_registry=plugin_registry
    )

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].script[0] = None

    with pytest.raises(TypeError, match="does not support item assignment"):
        config[0].script[0][0] = "x"


def test_indexed_ordered_dict(plugin_registry):
    idx_ordered_dict = ert3.config.IndexedOrderedDict(
        {"record_key_1": "record_value_1", "record_key_2": "record_value_2"}
    )
    assert idx_ordered_dict[0] == idx_ordered_dict["record_key_1"]


@pytest.mark.parametrize(
    "config,expected_transformation_cls,expected_transformation_attrs,direction",
    [
        pytest.param(
            [
                {
                    "name": "stage",
                    "input": [
                        {
                            "name": "some_record",
                            "transformation": {
                                "location": "params.json",
                            },
                        },
                    ],
                    "output": [],
                    "transportable_commands": [],
                    "script": [],
                }
            ],
            ert.data.CopyTransformation,
            [("location", pathlib.Path("params.json"))],
            ert.data.TransformationDirection.FROM_RECORD,
            id="copy_as_default_transformation_type",
        ),
        pytest.param(
            [
                {
                    "name": "stage",
                    "input": [
                        {
                            "name": "some_record",
                            "transformation": {
                                "type": "serialization",
                                "location": "params.json",
                            },
                        },
                    ],
                    "output": [],
                    "transportable_commands": [],
                    "script": [],
                }
            ],
            ert.data.SerializationTransformation,
            [("location", pathlib.Path("params.json")), ("mime", "application/json")],
            ert.data.TransformationDirection.FROM_RECORD,
            id="serialization",
        ),
        pytest.param(
            [
                {
                    "name": "stage",
                    "input": [
                        {
                            "name": "some_record",
                            "transformation": {
                                "type": "directory",
                                "location": "dir",
                            },
                        },
                    ],
                    "output": [],
                    "transportable_commands": [],
                    "script": [],
                }
            ],
            ert.data.TarTransformation,
            [("location", pathlib.Path("dir"))],
            ert.data.TransformationDirection.FROM_RECORD,
            id="directory",
        ),
        pytest.param(
            [
                {
                    "name": "stage",
                    "input": [],
                    "output": [
                        {
                            "name": "some_record",
                            "transformation": {
                                "type": "summary",
                                "smry_keys": ["*"],
                                "location": "summary",
                            },
                        },
                    ],
                    "transportable_commands": [],
                    "script": [],
                }
            ],
            ert.data.EclSumTransformation,
            [("location", pathlib.Path("summary")), ("smry_keys", ["*"])],
            ert.data.TransformationDirection.TO_RECORD,
            id="summary_output",
        ),
        pytest.param(
            [
                {
                    "name": "stage",
                    "input": [
                        {
                            "name": "some_record",
                            "transformation": {
                                "type": "summary",
                                "smry_keys": ["*"],
                                "location": "summary",
                            },
                        },
                    ],
                    "output": [],
                    "transportable_commands": [],
                    "script": [],
                }
            ],
            ert.data.EclSumTransformation,
            [("location", pathlib.Path("summary")), ("smry_keys", ["*"])],
            ert.data.TransformationDirection.FROM_RECORD,
            marks=pytest.mark.raises(
                exception=ValueError,
                match=".+cannot transform in direction: from_record",
                match_flags=(re.MULTILINE | re.DOTALL),
            ),
            id="summary_input",
        ),
        pytest.param(
            [
                {
                    "name": "stage",
                    "input": [
                        {
                            "name": "some_record",
                        },
                    ],
                    "output": [],
                    "transportable_commands": [],
                    "script": [],
                }
            ],
            None,
            [],
            ert.data.TransformationDirection.FROM_RECORD,
            id="missing transformation for unix step",
            marks=pytest.mark.raises(
                exception=ert.exceptions.ConfigValidationError,
                match=r".+io \'some_record\' had no transformation",
                match_flags=(re.MULTILINE | re.DOTALL),
            ),
        ),
    ],
)
def test_transformations(
    config: Dict[str, Any],
    expected_transformation_cls: Type[ert.data.RecordTransformation],
    expected_transformation_attrs: List[Tuple[str, Any]],
    direction: str,
    plugin_registry,
):
    stages_config = ert3.config.load_stages_config(
        config, plugin_registry=plugin_registry
    )
    if direction == ert.data.TransformationDirection.FROM_RECORD:
        input_ = stages_config[0].input[0]
        transformation = input_.get_transformation_instance()
    else:
        output = stages_config[0].output[0]
        transformation = output.get_transformation_instance()

    assert isinstance(transformation, expected_transformation_cls)
    for attr, expected_val in expected_transformation_attrs:
        assert getattr(transformation, attr) == expected_val


def test_required_plugged_in_configuration_errors():
    plugin_registry = ert3.config.ConfigPluginRegistry()
    plugin_registry.register_category(
        category="transformation",
        optional=False,
        base_config=ert3.config.plugins.TransformationConfigBase,
    )
    plugin_manager = ert3.plugins.ErtPluginManager(
        plugins=[ert3.config.plugins.implementations]
    )
    plugin_manager.collect(registry=plugin_registry)

    with pytest.raises(
        ert.exceptions.ConfigValidationError, match=r"transformation\n\s+field required"
    ):
        ert3.config.load_stages_config(
            [
                {
                    "name": "stage",
                    "input": [
                        {
                            "name": "some_record",
                        },
                    ],
                    "output": [],
                    "transportable_commands": [],
                    "script": [],
                }
            ],
            plugin_registry=plugin_registry,
        )


def test_optional_plugged_in_configuration():
    plugin_registry = ert3.config.ConfigPluginRegistry()
    plugin_registry.register_category(
        category="transformation",
        optional=True,
        base_config=ert3.config.plugins.TransformationConfigBase,
    )
    plugin_manager = ert3.plugins.ErtPluginManager(
        plugins=[ert3.config.plugins.implementations]
    )
    plugin_manager.collect(registry=plugin_registry)

    ert3.config.load_stages_config(
        [
            {
                "name": "stage",
                "input": [
                    {
                        "name": "some_record",
                    },
                ],
                "output": [],
                "function": "ert3.config:create_stages_config",
            }
        ],
        plugin_registry=plugin_registry,
    )


def test_no_plugged_configuration(base_unix_stage_config):
    # this attribute is added dynamically after _any_ create_stages_config call, so the
    # attribute is likely to exist. Removing it triggers the error this test needs.
    delattr(ert3.config._stages_config._Step, "_stageio_cls")

    with pytest.raises(
        RuntimeError,
        match="Step configuration must be obtained through 'create_stages_config'.",
    ):
        ert3.config.StagesConfig.parse_obj(base_unix_stage_config)
