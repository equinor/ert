import os
import string
from unittest.mock import patch

import pytest
from pydantic_core import ValidationError
from ruamel.yaml import YAML

from everest import ConfigKeys as CK
from everest import config_file_loader as loader
from everest.config import EverestConfig
from tests.everest.utils import relpath

mocked_root = relpath(os.path.join("test_data", "mocked_test_case"))


def test_load_yaml():
    config_file = os.path.join(mocked_root, "mocked_test_case.yml")
    configuration = loader.load_yaml(config_file)
    assert configuration is not None
    assert configuration[CK.DEFINITIONS] is not None


@patch.dict("os.environ", {"USER": "NO_USERNAME_ENV_SET"})
def test_load_yaml_preprocess():
    config_file = os.path.join(mocked_root, "mocked_test_case.yml")
    configuration = EverestConfig.load_file(config_file)
    username = os.environ.get("USER")
    folder = configuration.environment.simulation_folder
    assert f"simulations_{username}" == folder


def test_get_definitions():
    config_file = os.path.join(mocked_root, "mocked_test_case.yml")
    configuration = loader.load_yaml(config_file)
    definitions = loader._get_definitions(
        configuration=configuration,
        configpath=os.path.dirname(os.path.abspath(config_file)),
    )
    assert definitions is not None
    defs = {
        "case": "MOCKED_TEST_CASE",
        "configpath": os.path.dirname(os.path.abspath(config_file)),
        "runpath_file": "<RUNPATH_FILE>",
        "eclbase": "eclipse/ECL",
        "numeric_key": 1,
        "bool_key": True,
        "realization": "<GEO_ID>",
    }

    assert defs == definitions


def test_load_config_as_yaml():
    config_file = os.path.join(mocked_root, "mocked_test_case.yml")
    rendered_template = EverestConfig.load_file(config_file)
    assert rendered_template is not None


def test_configpath_in_defs(copy_mocked_test_data_to_tmp):
    config_file = "mocked_multi_batch.yml"
    config = EverestConfig.load_file(config_file)
    defs = {
        "numeric_key": 1,
        "bool_key": True,
        "eclbase": "MOCKED_TEST_CASE",
        "local_jobs_folder": os.path.join(os.getcwd(), "jobs"),
        "refcase_folder": os.path.join(os.getcwd(), "eclipse/refcase"),
        "spe1_datafile": os.path.join(os.getcwd(), "eclipse/refcase/SPE1.DATA"),
    }
    assert defs == config.definitions


def test_dependent_definitions(copy_mocked_test_data_to_tmp):
    config_file = "mocked_multi_batch.yml"
    with open(config_file, encoding="utf-8") as f:
        raw_config = YAML(typ="safe", pure=True).load(f)

    conseq_chars = zip(
        string.ascii_lowercase[:-1], string.ascii_lowercase[1:], strict=False
    )
    for c, cdef in [*list(conseq_chars), (string.ascii_lowercase[-1], "configpath")]:
        raw_config[CK.DEFINITIONS][c] = "r{{{{ {} }}}}".format(cdef)

    with open(config_file, "w", encoding="utf-8") as f:
        yaml = YAML(typ="safe", pure=True)
        yaml.indent = 2
        yaml.default_flow_style = False
        yaml.dump(raw_config, f)

    config = EverestConfig.load_file(config_file)

    defs = {
        "numeric_key": 1,
        "bool_key": True,
        "eclbase": "MOCKED_TEST_CASE",
        "local_jobs_folder": os.path.join(os.getcwd(), "jobs"),
        "refcase_folder": os.path.join(os.getcwd(), "eclipse/refcase"),
        "spe1_datafile": os.path.join(os.getcwd(), "eclipse/refcase/SPE1.DATA"),
    }
    defs.update({x: os.getcwd() for x in string.ascii_lowercase})
    assert defs == config.definitions


def test_dependent_definitions_value_error(copy_mocked_test_data_to_tmp):
    config_file = "mocked_multi_batch.yml"
    with open(config_file, encoding="utf-8") as f:
        raw_config = YAML(typ="safe", pure=True).load(f)

    raw_config[CK.DEFINITIONS]["a"] = "r{{ b }}"
    raw_config[CK.DEFINITIONS]["b"] = "r{{ a }}"

    with open(config_file, "w", encoding="utf-8") as f:
        yaml = YAML(typ="safe", pure=True)
        yaml.indent = 2
        yaml.default_flow_style = False
        yaml.dump(raw_config, f)
    with pytest.raises(ValueError):
        EverestConfig.load_file(config_file)


def test_load_empty_configuration(copy_mocked_test_data_to_tmp):
    with open("empty_config.yml", mode="w", encoding="utf-8") as fh:
        fh.writelines("")
    with pytest.raises(ValidationError, match="missing"):
        EverestConfig.load_file("empty_config.yml")


def test_load_invalid_configuration(copy_mocked_test_data_to_tmp):
    with open("invalid_config.yml", mode="w", encoding="utf-8") as fh:
        fh.writelines("asdf")
    with pytest.raises(ValidationError, match="missing"):
        EverestConfig.load_file("invalid_config.yml")
