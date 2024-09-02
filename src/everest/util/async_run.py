from threading import Thread


def async_run(function, on_finished=None, on_error=None):
    """Run a function in a separate thread.

    If the function raise an exception, on_error is called, with the
    exception as first argument.
    When the excecution is finished (either successfully or with an
    exception), on_finished is called with:
    - first argument is the value returned by the function
    - second argument is None if the function completed correctly, or
      it is the exception that stopeed the excecution
    """
    runner = _AsyncRunner(function, on_finished, on_error)
    runner.start()


class _AsyncRunner(Thread):
    def __init__(self, function=None, on_finished=None, on_error=None):
        super(_AsyncRunner, self).__init__()
        self._function = function
        self._on_finished = on_finished
        self._on_error = on_error

    def run(self):
        ret = None
        error = None
        try:
            ret = self._function()
        except Exception as e:
            error = e
            if self._on_error is not None:
                self._on_error(e)
        finally:
            if self._on_finished is not None:
                self._on_finished(ret, error)
