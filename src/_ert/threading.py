from __future__ import annotations

import logging
import os
import signal
import threading
from threading import Thread as _Thread
from types import FrameType
from typing import Optional

logger = logging.getLogger(__name__)


class ErtThreadError(Exception):
    def __init__(self, exception: BaseException, thread: _Thread) -> None:
        super().__init__(repr(exception))
        self._exception = exception
        self._thread = thread

    @property
    def exception(self) -> BaseException:
        return self._exception

    def __str__(self) -> str:
        return f"{self._exception} in thread '{self._thread.name}'"


class ErtThread(_Thread):
    def run(self) -> None:
        try:
            super().run()
        except BaseException as exc:
            logger.error(str(exc), exc_info=exc)

            # Re-raising this exception on main thread can have unknown
            # repercussions in production. Potentially, an unnecessary thread
            # was dying due to an exception and we didn't care, but with this
            # change this would bring down all of Ert. We take the conservative
            # approach and make re-raising optional, and enable it only for
            # the test suite.
            if os.environ.get("_ERT_THREAD_RAISE", ""):
                _raise_on_main_thread(exc)


_current_exception: Optional[BaseException] = None


def _raise_on_main_thread(exception: BaseException) -> None:
    if threading.main_thread() is threading.current_thread():
        raise exception

    global _current_exception  # noqa: PLW0603
    _current_exception = ErtThreadError(exception, threading.current_thread())

    # Send a signal to ourselves. On POSIX, the signal is explicitly handled on
    # the main thread, thus this is a way for us to inform the main thread that
    # something has happened. SIGUSR1 is a user-defined signal. Unlike SIGABRT
    # where the handler must eventually abort, we are allowed to continue
    # operating as normal after handling SIGUSR1.
    os.kill(os.getpid(), signal.SIGUSR1)


def _handler(signum: int, frametype: FrameType | None) -> None:
    global _current_exception  # noqa: PLW0603
    if not _current_exception:
        return
    current_exception, _current_exception = _current_exception, None
    raise current_exception


signal.signal(signal.SIGUSR1, _handler)
