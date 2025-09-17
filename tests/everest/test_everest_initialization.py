from pathlib import Path
from textwrap import dedent
from unittest import mock

import pytest

from ert.plugins import ErtPluginContext, ErtRuntimePlugins
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig


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
        "ert.plugins.plugin_manager.ErtRuntimePlugins",
        return_value=ErtRuntimePlugins(
            environment_variables={"HOW_MANY_CPU": "<NUM_CPU>"}
        ),
    ):
        config = EverestConfig.with_defaults()

        with ErtPluginContext() as runtime_plugins:
            everest_run_model = EverestRunModel.create(
                config, runtime_plugins=runtime_plugins
            )

        assert ("<NUM_CPU>", "1") in everest_run_model.substitutions.items()
        assert everest_run_model.env_vars["HOW_MANY_CPU"] == "1"
