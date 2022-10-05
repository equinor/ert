from decorator import decorator


@decorator
def plugin_response(_func=None, plugin_name="", *args, **kwargs):

    if _func is None:
        return None

    response = _func(*args, **kwargs)
    return (
        PluginResponse(response, PluginMetadata(plugin_name, _func.__name__))
        if response is not None
        else None
    )


class PluginResponse:
    def __init__(self, data, plugin_metadata):
        self.data = data
        self.plugin_metadata = plugin_metadata


class PluginMetadata:
    def __init__(self, plugin_name, function_name):
        self.plugin_name = plugin_name
        self.function_name = function_name
