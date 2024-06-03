import os

import pytest
from everest.config import EverestConfig
from everest.suite import _EverestWorkflow

from tests.utils import relpath, tmp

NO_PROJECT_RES = (
    os.environ.get("NO_PROJECT_RES", False),
    "Skipping tests when no access to /project/res",
)


@pytest.mark.skipif(NO_PROJECT_RES[0], reason=NO_PROJECT_RES[1])
def test_init_no_project_res():
    root_dir = relpath("..", "examples", "egg")
    config_file = os.path.join("everest", "model", "config.yml")
    with tmp(root_dir):
        config_dict = EverestConfig.load_file(config_file)
        _EverestWorkflow(config_dict)


def test_init():
    root_dir = relpath("test_data", "mocked_test_case")
    config_file = os.path.join("mocked_test_case.yml")
    with tmp(root_dir):
        config_dict = EverestConfig.load_file(config_file)
        _EverestWorkflow(config_dict)


def test_no_config_init():
    with pytest.raises(AttributeError):
        _EverestWorkflow(None)  # type: ignore
    with pytest.raises(AttributeError):
        _EverestWorkflow("Frozen bananas")  # type: ignore
