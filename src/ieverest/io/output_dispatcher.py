import inspect
import logging


class OutputDispatcher(object):
    """Forward output requests to multiple output channels"""

    def __init__(self, output_channels={}):  # noqa B006
        super(OutputDispatcher, self).__init__()
        self.output_channels = output_channels

    def add_output_channel(self, channel, channel_id):
        self.output_channels[channel_id] = channel

    def _get_out_channels(self, channel_ids=None):
        if channel_ids is None:
            return self.output_channels
        return {k: ch for k, ch in self.output_channels.items() if k in channel_ids}

    def _forward_to_output(self, func_name, message, channels=None, **kwargs):
        for k, ch in self._get_out_channels(channels).items():
            try:
                func = getattr(ch, func_name)
                args, _, has_kwargs, *_ = inspect.getfullargspec(func)
                if has_kwargs:
                    func(message, **kwargs)
                else:
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in args}
                    func(message, **filtered_kwargs)
            except:
                logging.getLogger().debug(
                    "Can't find function {} on channel {}".format(func_name, k)
                )

    def debug(self, message, channels=None, **kwargs):
        self._forward_to_output("debug", message, channels, **kwargs)

    def info(self, message, channels=None, **kwargs):
        self._forward_to_output("info", message, channels, **kwargs)

    def warning(self, message, channels=None, **kwargs):
        self._forward_to_output("warning", message, channels, **kwargs)

    def _error(self, message, channels=None, **kwargs):
        self._forward_to_output("_error", message, channels, **kwargs)

    def critical(self, message, channels=None, **kwargs):
        self._forward_to_output("critical", message, channels, **kwargs)
