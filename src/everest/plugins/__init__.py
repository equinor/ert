import pluggy

from everest.plugins.plugin_response import plugin_response
from everest.strings import EVEREST

hookimpl = pluggy.HookimplMarker(EVEREST)
hookspec = pluggy.HookspecMarker(EVEREST)


__all__ = ["EVEREST", "plugin_response"]
