import os
from pathlib import Path
from textwrap import dedent
from unittest import mock

import pytest

from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig

from .utils import skipif_no_everest_models


@skipif_no_everest_models
@pytest.mark.requires_eclipse
def test_init_no_project_res(copy_eightcells_test_data_to_tmp):
    config_file = os.path.join("everest", "model", "config.yml")
    config = EverestConfig.load_file(config_file)
    EverestRunModel.create(config)


def test_no_config_init():
    with pytest.raises(AttributeError):
        EverestRunModel.create(None)

    with pytest.raises(AttributeError):
        EverestRunModel.create("Frozen bananas")


def test_site_config_with_substitutions(monkeypatch, change_to_tmpdir):
    # set up siteconfig
    test_site_config = Path("test_site_config.ert")
    test_site_config.write_text(
        dedent("""
        SETENV HOW_MANY_CPU <NUM_CPU>
        """),
        encoding="utf-8",
    )

    with mock.patch(
        "ert.config.ert_config.site_config_location", return_value=str(test_site_config)
    ):
        config = EverestConfig.with_defaults()
        everest_run_model = EverestRunModel.create(config)

        assert ("<NUM_CPU>", "1") in everest_run_model.substitutions.items()
        assert everest_run_model.env_vars["HOW_MANY_CPU"] == "1"
