import os

import pytest

from everest.config import EverestConfig
from everest.suite import _EverestWorkflow

NO_PROJECT_RES = (
    os.environ.get("NO_PROJECT_RES", False),
    "Skipping tests when no access to /project/res",
)


@pytest.mark.skipif(NO_PROJECT_RES[0], reason=NO_PROJECT_RES[1])
def test_init_no_project_res(copy_egg_test_data_to_tmp):
    config_file = os.path.join("everest", "model", "config.yml")
    config_dict = EverestConfig.load_file(config_file)
    _EverestWorkflow(config_dict)


def test_init(copy_mocked_test_data_to_tmp):
    config_file = os.path.join("mocked_test_case.yml")
    config_dict = EverestConfig.load_file(config_file)
    _EverestWorkflow(config_dict)


def test_no_config_init():
    with pytest.raises(AttributeError):
        _EverestWorkflow(None)  # type: ignore
    with pytest.raises(AttributeError):
        _EverestWorkflow("Frozen bananas")  # type: ignore
