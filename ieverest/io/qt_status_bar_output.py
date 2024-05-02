from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING


class QtStatusBarOut(object):
    """Support output requests through Qt Status Bar"""

    def __init__(self, qt_status_bar, log_level=INFO, default_timeout=2000):
        super(QtStatusBarOut, self).__init__()
        self._bar = qt_status_bar
        self.log_level = log_level
        self.default_timeout = default_timeout

    def _show_msg(self, level, message, timeout=None, force=False):
        """Show a message on the status bar.

        The message is shown iff its level is greater than self.log_level or if
        force is True. If timeout is a positive number, the message disappear
        after the given timeout (in ms). If timeout is None, the default_timeout
        is used. If timeout is -1, no timeout is set.
        """
        if level < self.log_level and not force:
            return
        if timeout is None:
            timeout = self.default_timeout
        if timeout < 0:
            self._bar.showMessage(message)
        else:
            self._bar.showMessage(message, timeout)

    def critical(self, message, timeout=None, force=False):
        self._show_msg(CRITICAL, message, timeout, force)

    def _error(self, message, timeout=None, force=False):
        self._show_msg(ERROR, message, timeout, force)

    def warning(self, message, timeout=None, force=False):
        self._show_msg(WARNING, message, timeout, force)

    def info(self, message, timeout=None, force=False):
        self._show_msg(INFO, message, timeout, force)

    def debug(self, message, timeout=None, force=False):
        self._show_msg(DEBUG, message, timeout, force)
