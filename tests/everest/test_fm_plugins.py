import logging
import re
from collections.abc import Callable, Iterator, Sequence
from itertools import chain

import pluggy
import pytest
from pydantic import BaseModel

from ert import ForwardModelStepPlugin
from ert.config import ErtConfig
from ert.plugins import ErtPluginContext
from everest.plugins import hook_impl, hook_specs, hookimpl
from everest.strings import EVEREST


class MockPluginManager(pluggy.PluginManager):
    """A testing plugin manager"""

    def __init__(self) -> None:
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
    jobs = everest_models.forward_models.get_forward_models()
    assert bool(jobs)
    with ErtPluginContext() as ctx:
        for job in jobs:
            job_class = ErtConfig.with_plugins(
                ctx
            ).PREINSTALLED_FORWARD_MODEL_STEPS.get(job)
            assert job_class is not None
            assert isinstance(job_class, ForwardModelStepPlugin)


def test_multiple_plugins(plugin_manager):
    SCHEMAS = [{"job1": 1}, {"job2": 2}]

    class Plugin1:
        @hookimpl
        def get_forward_models_schemas(self):
            return [SCHEMAS[0]]

    class Plugin2:
        @hookimpl
        def get_forward_models_schemas(self):
            return [SCHEMAS[1]]

    pm = plugin_manager(Plugin1(), Plugin2())

    jobs = list(chain.from_iterable(pm.hook.get_forward_models_schemas()))
    for value in SCHEMAS:
        assert value in jobs


def test_parse_forward_model_schema(plugin_manager):
    class Model(BaseModel):
        content: str

    class Plugin:
        @hookimpl
        def parse_forward_model_schema(self, path: str, schema: type[BaseModel]):
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

    assert next(
        chain.from_iterable(
            pm.hook.lint_forward_model(
                job="some_forward_model",
                args=["--config", "path/to/somewhere"],
            )
        )
    ) == (
        "Mocked error message: some_forward_model -> ['--config', 'path/to/somewhere']"
    )


def test_add_logging_handle(plugin_manager):
    handle = logging.StreamHandler()

    class Plugin:
        @hookimpl
        def add_log_handle_to_root(self):
            return handle

    pm = plugin_manager(Plugin())
    assert pm.hook.add_log_handle_to_root() == [handle]


def test_logging_from_plugin(caplog, plugin_manager):
    handle = logging.StreamHandler()

    logging.getLogger("my.test").addHandler(handle)

    class Plugin:
        @hookimpl
        def add_log_handle_to_root(self):
            return handle

        def log(self):
            logging.getLogger("my.plugin").info("Hello from plugin")

    pm = plugin_manager(Plugin())
    pm.hook.add_log_handle_to_root()

    caplog.set_level(logging.DEBUG)

    logging.getLogger("my.test").debug("Hello from my test")
    Plugin().log()

    assert re.search(r"DEBUG.*my\.test:.*Hello from my test", caplog.text)
    assert re.search(r"INFO.*my\.plugin:.*Hello from plugin", caplog.text)
