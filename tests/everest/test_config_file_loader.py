import os
import string
from unittest.mock import patch

import pytest

from everest import config_file_loader as loader
from everest.config import EverestConfig
from everest.config.everest_config import EverestValidationError
from tests.everest.utils import everest_config_with_defaults


def test_load_yaml(tmp_path):
    config_file_path = tmp_path / "config.yml"

    initial_config = everest_config_with_defaults(definitions={})

    initial_config.write_to_file(config_file_path)

    loaded_config = EverestConfig.load_file(str(config_file_path))

    assert loaded_config is not None
    assert loaded_config.definitions is not None


@patch.dict("os.environ", {"USER": "NO_USERNAME_ENV_SET"})
def test_load_yaml_preprocess(tmp_path):
    config_file = tmp_path / "config.yml"

    initial_config = everest_config_with_defaults(
        environment={"simulation_folder": "simulations_r{{ os.USER }}"}
    )

    initial_config.write_to_file(config_file)

    config = EverestConfig.load_file(str(config_file))

    assert config.environment.simulation_folder == "simulations_NO_USERNAME_ENV_SET"


def test_get_definitions(tmp_path):
    initial_config = everest_config_with_defaults(
        definitions={
            "case": "MOCKED_TEST_CASE",
            "eclbase": "eclipse/ECL",
            "numeric_key": 1,
            "bool_key": True,
        }
    )

    config_file = tmp_path / "config.yml"

    initial_config.write_to_file(config_file)

    config = loader.load_yaml(config_file)
    definitions = loader._get_definitions(
        configuration=config,
        configpath=os.path.dirname(os.path.abspath(config_file)),
    )

    assert definitions is not None

    expected_definitions = {
        "case": "MOCKED_TEST_CASE",
        "configpath": os.path.dirname(os.path.abspath(config_file)),
        "runpath_file": "<RUNPATH_FILE>",
        "eclbase": "eclipse/ECL",
        "numeric_key": 1,
        "bool_key": True,
        "realization": "<REALIZATION_ID>",
    }

    assert definitions == expected_definitions


def test_load_config_as_yaml(tmp_path):
    config_file = tmp_path / "config.yml"

    initial_config = everest_config_with_defaults(definitions={"case": "MINIMAL_CASE"})

    initial_config.write_to_file(config_file)

    assert EverestConfig.load_file(str(config_file)) is not None


def test_configpath_in_defs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    current_working_directory = os.getcwd()

    definitions_for_yaml_creation = {
        "local_jobs_folder": "r{{ configpath }}/jobs",
    }

    config_file_path = tmp_path / "config.yml"

    config = everest_config_with_defaults(definitions=definitions_for_yaml_creation)
    config.write_to_file(config_file_path)

    loaded_config = EverestConfig.load_file(str(config_file_path))

    expected_definitions_after_load = {
        "local_jobs_folder": os.path.join(current_working_directory, "jobs"),
    }

    assert expected_definitions_after_load == loaded_config.definitions


def test_dependent_definitions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    current_working_directory = os.getcwd()

    definitions_for_initial_config_object = {
        "numeric_key": 1,
        "bool_key": True,
        "local_jobs_folder": "r{{ configpath }}/jobs",
    }

    all_chars = string.ascii_lowercase
    for i in range(len(all_chars) - 1):
        current_char = all_chars[i]
        next_char = all_chars[i + 1]
        definitions_for_initial_config_object[current_char] = f"r{{{{ {next_char} }}}}"

    definitions_for_initial_config_object[all_chars[-1]] = "r{{ configpath }}"

    initial_config_object = everest_config_with_defaults(
        definitions=definitions_for_initial_config_object
    )

    config_file_path = tmp_path / "config.yml"
    initial_config_object.write_to_file(config_file_path)

    loaded_config = EverestConfig.load_file(str(config_file_path))

    expected_defs = {
        "numeric_key": 1,
        "bool_key": True,
        "local_jobs_folder": os.path.join(current_working_directory, "jobs"),
        **dict.fromkeys(all_chars, current_working_directory),
    }

    assert expected_defs == loaded_config.definitions


def test_dependent_definitions_value_error(tmp_path):
    config_file_path = tmp_path / "config.yml"

    definitions_with_circular_dependency = {
        "a": "r{{ b }}",
        "b": "r{{ a }}",
        "numeric_key": 1,
        "bool_key": True,
        "local_jobs_folder": "{{ configpath }}/jobs",
    }

    initial_config_object = everest_config_with_defaults(
        definitions=definitions_with_circular_dependency
    )
    initial_config_object.write_to_file(config_file_path)

    with pytest.raises(ValueError):
        EverestConfig.load_file(str(config_file_path))


def test_load_empty_configuration(tmp_path):
    (tmp_path / "config.yml").touch()
    with pytest.raises(EverestValidationError, match="missing"):
        EverestConfig.load_file(tmp_path / "config.yml")


def test_load_invalid_configuration(tmp_path):
    (tmp_path / "config.yml").write_text("asdf", encoding="utf-8")
    with pytest.raises(EverestValidationError, match="missing"):
        EverestConfig.load_file(tmp_path / "config.yml")
