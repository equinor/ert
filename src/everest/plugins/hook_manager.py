import pluggy

from everest.plugins import hook_impl, hook_specs
from everest.strings import EVEREST


class EverestPluginManager(pluggy.PluginManager):
    def __init__(self, plugins=None):
        super(EverestPluginManager, self).__init__(EVEREST)
        self.add_hookspecs(hook_specs)
        if plugins is None:
            self.register(hook_impl)
            self.load_setuptools_entrypoints(EVEREST)
        else:
            for plugin in plugins:
                self.register(plugin)
