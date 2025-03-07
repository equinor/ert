import os
from pathlib import Path
from textwrap import dedent

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


def test_site_config_with_substitutions(monkeypatch, copy_math_func_test_data_to_tmp):
    # set up siteconfig
    test_site_config = Path("test_site_config.ert")
    test_site_config.write_text(
        dedent("""
        SETENV HOW_MANY_CPU <NUM_CPU>
        """),
        encoding="utf-8",
    )
    monkeypatch.setenv("ERT_SITE_CONFIG", str(test_site_config))

    config = EverestConfig.load_file("config_minimal.yml")
    everest_run_model = EverestRunModel.create(config)

    assert ("<NUM_CPU>", "1") in everest_run_model._substitutions.items()
    assert everest_run_model._env_vars["HOW_MANY_CPU"] == "1"
