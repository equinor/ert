from typing import Any, Dict

import pluggy

from everest.plugins import hook_impl, hook_specs
from everest.strings import EVEREST


class EverestPluginManager(pluggy.PluginManager):
    def __init__(self, plugins=None) -> None:
        super(EverestPluginManager, self).__init__(EVEREST)
        self.add_hookspecs(hook_specs)
        if plugins is None:
            self.register(hook_impl)
            self.load_setuptools_entrypoints(EVEREST)
        else:
            for plugin in plugins:
                self.register(plugin)

    def get_documentation(self) -> Dict[str, Any]:
        docs = self.hook.get_forward_model_documentations()
        return {k: v for d in docs for k, v in d.items()} if docs else {}
