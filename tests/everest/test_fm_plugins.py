import logging
from itertools import chain
from typing import Callable, Iterator, Sequence, Type

import pluggy
import pytest
from pydantic import BaseModel

from everest.plugins import hook_impl, hook_specs, hookimpl
from everest.strings import EVEREST
from everest.util.forward_models import collect_forward_models


class MockPluginManager(pluggy.PluginManager):
    """A testing plugin manager"""

    def __init__(self):
        super().__init__(EVEREST)
        self.add_hookspecs(hook_specs)


@pytest.fixture
def plugin_manager() -> Iterator[Callable[..., MockPluginManager]]:
    pm = MockPluginManager()

    def register_plugin_hooks(*plugins) -> MockPluginManager:
        if not plugins:
            pm.register(hook_impl)
            pm.load_setuptools_entrypoints(EVEREST)
        else:
            for plugin in plugins:
                pm.register(plugin)
        return pm

    yield register_plugin_hooks


def test_jobs():
    for job in collect_forward_models():
        assert "name" in job
        assert "path" in job


def test_everest_models_jobs(plugin_manager):
    pytest.importorskip("everest_models")
    pm = plugin_manager()
    assert any(
        hook.plugin_name.startswith(EVEREST)
        for hook in pm.hook.get_forward_models.get_hookimpls()
    )


def test_multiple_plugins(plugin_manager):
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

    pm = plugin_manager(Plugin1(), Plugin2())

    jobs = list(chain.from_iterable(pm.hook.get_forward_models()))
    for value in _JOBS:
        assert value in jobs


def test_parse_forward_model_schema(plugin_manager):
    class Model(BaseModel):
        content: str

    class Plugin:
        @hookimpl
        def parse_forward_model_schema(self, path: str, schema: Type[BaseModel]):
            return schema.model_validate({"content": path})

    pm = plugin_manager(Plugin())

    assert next(
        chain.from_iterable(
            pm.hook.parse_forward_model_schema(path="/path/to/config.yml", schema=Model)
        )
    ) == ("content", "/path/to/config.yml")


def test_lint_forward_model_hook(plugin_manager):
    class Plugin:
        @hookimpl
        def lint_forward_model(self, job: str, args: Sequence[str]):
            return [[f"Mocked error message: {job} -> {args}"]]

    pm = plugin_manager(Plugin())

    assert (
        next(
            chain.from_iterable(
                pm.hook.lint_forward_model(
                    job="some_forward_model",
                    args=["--config", "path/to/somewhere"],
                )
            )
        )
        == "Mocked error message: some_forward_model -> ['--config', 'path/to/somewhere']"
    )


def test_add_logging_handle(plugin_manager):
    handle = logging.StreamHandler()

    class Plugin:
        @hookimpl
        def add_log_handle_to_root(self):
            return handle

    pm = plugin_manager(Plugin())
    assert pm.hook.add_log_handle_to_root() == [handle]
