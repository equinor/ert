from __future__ import annotations

import logging
import os
import signal
import threading
import traceback
from collections.abc import Callable, Iterable
from threading import Thread as _Thread
from types import FrameType
from typing import Any

logger = logging.getLogger(__name__)


_current_exception: ErtThreadError | None = None
_can_raise = False


class ErtThreadError(Exception):
    def __init__(
        self, exception: BaseException, thread: _Thread, err_traceback: str
    ) -> None:
        super().__init__(repr(exception))
        self._exception = exception
        self._thread = thread
        self._err_traceback = err_traceback

    @property
    def exception(self) -> BaseException:
        return self._exception

    def __str__(self) -> str:
        return (
            f"{self._exception} in thread '{self._thread.name}'\n {self._err_traceback}"
        )

    def __repr__(self) -> str:
        return str(self)


class ErtThread(_Thread):
    def __init__(
        self,
        target: Callable[..., Any],
        name: str | None = None,
        args: Iterable[Any] = (),
        *,
        daemon: bool | None = None,
        should_raise: bool = True,
    ) -> None:
        super().__init__(target=target, name=name, args=args, daemon=daemon)
        self._should_raise = should_raise

    def run(self) -> None:
        try:
            super().run()
        except BaseException as exc:
            err_traceback = str(traceback.format_exc())
            logger.error(err_traceback, exc_info=exc)
            if _can_raise and self._should_raise:
                _raise_on_main_thread(exc)


def _raise_on_main_thread(exception: BaseException) -> None:
    err_traceback = str(traceback.format_exc())
    global _current_exception  # noqa: PLW0603
    _current_exception = ErtThreadError(
        exception, threading.current_thread(), err_traceback
    )

    # Send a signal to ourselves. On POSIX, the signal is explicitly handled on
    # the main thread, thus this is a way for us to inform the main thread that
    # something has happened. SIGUSR1 is a user-defined signal. Unlike SIGABRT
    # where the handler must eventually abort, we are allowed to continue
    # operating as normal after handling SIGUSR1.
    os.kill(os.getpid(), signal.SIGUSR1)


def _handler(signum: int, frametype: FrameType | None) -> None:
    global _current_exception
    if not _current_exception:
        return
    current_exception, _current_exception = _current_exception, None
    raise current_exception


def set_signal_handler() -> None:
    global _can_raise  # noqa: PLW0603
    if _can_raise:
        return

    signal.signal(signal.SIGUSR1, _handler)
    _can_raise = True
