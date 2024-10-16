from typing import Generic, TypeVar

from decorator import decorator

T = TypeVar("T")


@decorator
def plugin_response(func, plugin_name="", *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
    response = func(*args, **kwargs)
    return (
        PluginResponse(response, PluginMetadata(plugin_name, func.__name__))
        if response is not None
        else None
    )


class PluginMetadata:
    def __init__(self, plugin_name, function_name):
        self.plugin_name = plugin_name
        self.function_name = function_name


class PluginResponse(Generic[T]):
    def __init__(self, data: T, plugin_metadata: PluginMetadata) -> None:
        self.data = data
        self.plugin_metadata = plugin_metadata
