from unittest import mock

import pytest

from ert.base_model_context import use_runtime_plugins
from ert.plugins import ErtRuntimePlugins, get_site_plugins
from ert.run_models.everest_run_model import EverestRunModel
from everest.config import EverestConfig


def test_no_config_init():
    with pytest.raises(AttributeError):
        EverestRunModel.create(None)

    with pytest.raises(AttributeError):
        EverestRunModel.create("Frozen bananas")


def test_site_config_with_substitutions(monkeypatch, change_to_tmpdir):
    def runtime_plugins_with_cpu_override(**kwargs):
        return ErtRuntimePlugins(
            **(
                kwargs
                | {"queue_options": None}
                | {"environment_variables": {"HOW_MANY_CPU": "<NUM_CPU>"}}
            )
        )

    with mock.patch(
        "ert.plugins.plugin_manager.ErtRuntimePlugins",
        side_effect=runtime_plugins_with_cpu_override,
    ):
        config = EverestConfig.with_defaults()

        runtime_plugins = get_site_plugins()
        with use_runtime_plugins(runtime_plugins):
            everest_run_model = EverestRunModel.create(
                config, runtime_plugins=runtime_plugins
            )

        assert ("<NUM_CPU>", "1") in everest_run_model.substitutions.items()
        assert everest_run_model.env_vars["HOW_MANY_CPU"] == "1"
