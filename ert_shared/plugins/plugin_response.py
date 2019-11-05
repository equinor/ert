import functools


def plugin_response(plugin_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper_decorator(*args, **kwargs):
            return PluginResponse(func(*args, **kwargs), PluginMetadata(plugin_name, func.__name__))
        return wrapper_decorator
    return decorator


class PluginResponse:
    def __init__(self, data, plugin_metadata):
        self.data = data
        self.plugin_metadata = plugin_metadata

class PluginMetadata:
    def __init__(self, plugin_name, function_name):
        self.plugin_name = plugin_name
        self.function_name = function_name
