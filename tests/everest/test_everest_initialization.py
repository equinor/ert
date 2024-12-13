import os

import pytest

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig

from .utils import skipif_no_everest_models


@skipif_no_everest_models
@pytest.mark.requires_eclipse
def test_init_no_project_res(copy_egg_test_data_to_tmp):
    config_file = os.path.join("everest", "model", "config.yml")
    config = EverestConfig.load_file(config_file)
    EverestRunModel.create(config)


def test_init(copy_mocked_test_data_to_tmp):
    config_file = os.path.join("mocked_test_case.yml")
    config = EverestConfig.load_file(config_file)
    EverestRunModel.create(config)


def test_no_config_init():
    with pytest.raises(AttributeError):
        EverestRunModel.create(None)

    with pytest.raises(AttributeError):
        EverestRunModel.create("Frozen bananas")
