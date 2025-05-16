import logging
from typing import Any

import pluggy

from ert.trace import add_span_processor
from everest.plugins import hook_impl, hook_specs
from everest.strings import EVEREST


class EverestPluginManager(pluggy.PluginManager):
    def __init__(self, plugins: list[Any] | None = None) -> None:
        super().__init__(EVEREST)
        self.add_hookspecs(hook_specs)
        if plugins is None:
            self.register(hook_impl)
            self.load_setuptools_entrypoints(EVEREST)
        else:
            for plugin in plugins:
                self.register(plugin)

    def get_documentation(self) -> dict[str, Any]:
        docs = self.hook.get_forward_model_documentations()
        return {k: v for d in docs for k, v in d.items()} if docs else {}

    def add_log_handle_to_root(self) -> None:
        root_logger = logging.getLogger()
        for handler in self.hook.add_log_handle_to_root():
            root_logger.addHandler(handler)

    def add_span_processor_to_trace_provider(self) -> None:
        span_processors = self.hook.add_span_processor()
        for span_processor in span_processors:
            add_span_processor(span_processor)
