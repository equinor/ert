import logging
from itertools import chain
from typing import Callable, Iterator, Sequence, Type

import pluggy
import pytest
from pydantic import BaseModel

from ert import ForwardModelStepPlugin
from everest.config import EverestConfig
from everest.plugins import hook_impl, hook_specs, hookimpl
from everest.simulator.everest_to_ert import everest_to_ert_config
from everest.strings import EVEREST
from tests.everest.utils import relpath

SNAKE_CONFIG_PATH = relpath("test_data/snake_oil/everest/model/snake_oil.yml")


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


def test_everest_models_jobs():
    everest_models = pytest.importorskip("everest_models")
    ert_config = everest_to_ert_config(EverestConfig.load_file(SNAKE_CONFIG_PATH))
    jobs = everest_models.forward_models.get_forward_models()
    assert bool(jobs)
    for job in jobs:
        job_class = ert_config.installed_forward_model_steps.get(job)
        assert job_class is not None
        assert isinstance(job_class, ForwardModelStepPlugin)


def test_multiple_plugins(plugin_manager):
    _SCHEMAS = [{"job1": 1}, {"job2": 2}]

    class Plugin1:
        @hookimpl
        def get_forward_models_schemas(self):
            return [_SCHEMAS[0]]

    class Plugin2:
        @hookimpl
        def get_forward_models_schemas(self):
            return [_SCHEMAS[1]]

    pm = plugin_manager(Plugin1(), Plugin2())

    jobs = list(chain.from_iterable(pm.hook.get_forward_models_schemas()))
    for value in _SCHEMAS:
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
