import logging
from itertools import chain

import pytest
from everest.plugins import hookimpl
from everest.plugins.hook_manager import EverestPluginManager
from everest.util.forward_models import collect_forward_models


def test_jobs():
    for job in collect_forward_models():
        assert "name" in job
        assert "path" in job


def test_everest_models_jobs():
    pytest.importorskip("everest_models")
    pm = EverestPluginManager()
    assert any(
        hook.plugin_name.startswith("everest-models")
        for hook in pm.hook.get_forward_models.get_hookimpls()
    )


def test_multiple_plugins():
    _JOBS = [
        {"name": "job1", "path": "/some/path1"},
        {"name": "job2", "path": "/some/path2"},
    ]

    class Plugin1:
        @hookimpl
        def get_forward_models(self):
            return [_JOBS[0]]

    class Plugin2:
        @hookimpl
        def get_forward_models(self):
            return [_JOBS[1]]

    pm = EverestPluginManager()
    pm.register(Plugin1())
    pm.register(Plugin2())

    jobs = list(chain.from_iterable(pm.hook.get_forward_models()))
    for value in _JOBS:
        assert value in jobs


def test_add_logging_handle():
    pm = EverestPluginManager()
    hook_log_handles = pm.hook.add_log_handle_to_root()
    handle = logging.StreamHandler()

    class Plugin1:
        @hookimpl
        def add_log_handle_to_root(self):
            return handle

    if not hook_log_handles:
        pm.register(Plugin1())
        assert pm.hook.add_log_handle_to_root() == [handle]
    else:
        assert len(hook_log_handles) == 1
        assert type(hook_log_handles[0]).__name__ == "FMUAzureLogHandler"
