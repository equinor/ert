"""Capturing what workflow jobs write to ``sys.stdout``/``sys.stderr``.

:func:`contextlib.redirect_stdout` is deliberately not used here. It swaps
``sys.stdout`` process-wide and hands the stream to a single writer, whereas
capturing workflow job output has to

* pass writes on to the stream it replaced, so output still reaches the
  terminal as the job runs rather than only appearing once it is over,
* keep each thread's writes apart, since workflow runners each run their jobs
  on a thread of their own and a job must not pick up output another thread
  happened to write, and
* survive overlapping captures, restoring the original stream only once the
  last capture is done and only if nothing else has replaced it since.
"""

from __future__ import annotations

import contextlib
import io
import sys
import threading
from collections.abc import Iterator
from typing import Any, TextIO, override


class _CaptureProxy(io.TextIOBase):
    """Stands in for ``sys.stdout``/``sys.stderr`` while workflow jobs run.

    Internal jobs print straight to ``sys.stdout``/``sys.stderr``, so replacing
    those streams is the only way to get hold of what they write. Everything
    written is passed on to the stream this proxy replaced, so output still
    reaches the terminal.

    Capture is per-thread: a thread collects only what it writes itself, so
    workflow jobs running concurrently do not pick up each other's output.
    Threads that are not capturing are unaffected beyond the forwarding.
    """

    def __init__(self, stream: TextIO) -> None:
        super().__init__()
        self.wrapped_stream = stream
        self._local = threading.local()
        self._active_captures = 0

    @property
    def _buffers(self) -> list[io.StringIO]:
        buffers: list[io.StringIO] | None = getattr(self._local, "buffers", None)
        if buffers is None:
            buffers = []
            self._local.buffers = buffers
        return buffers

    @contextlib.contextmanager
    def capture(self) -> Iterator[io.StringIO]:
        """Record what the calling thread writes for the duration of the block."""
        buffer = io.StringIO()
        buffers = self._buffers
        buffers.append(buffer)
        try:
            yield buffer
        finally:
            buffers.remove(buffer)

    @override
    def write(self, s: str, /) -> int:
        for buffer in self._buffers:
            buffer.write(s)
        return self.wrapped_stream.write(s)

    @override
    def flush(self) -> None:
        self.wrapped_stream.flush()

    @override
    def close(self) -> None:
        """Closing is ignored, as the wrapped stream outlives the capture."""

    @override
    def fileno(self) -> int:
        return self.wrapped_stream.fileno()

    @override
    def isatty(self) -> bool:
        return self.wrapped_stream.isatty()

    @override
    def writable(self) -> bool:
        # IOBase defaults to False, so this one is load-bearing. readable() and
        # seekable() are left to IOBase, which already reports False.
        return True

    @property
    @override
    def encoding(self) -> str:  # type: ignore[override]
        # TextIOBase defines encoding, errors and newlines as descriptors
        # returning None, so __getattr__ is never consulted for them.
        return getattr(self.wrapped_stream, "encoding", "utf-8")

    @property
    @override
    def errors(self) -> str | None:  # type: ignore[override]
        return getattr(self.wrapped_stream, "errors", None)

    @property
    @override
    def newlines(self) -> Any:  # type: ignore[override]
        return getattr(self.wrapped_stream, "newlines", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["wrapped_stream"], name)


_capture_lock = threading.Lock()


@contextlib.contextmanager
def capturing(stream_name: str) -> Iterator[io.StringIO]:
    # Record what the calling thread writes to ``sys.<stream_name>``
    proxy: _CaptureProxy
    with _capture_lock:
        stream = getattr(sys, stream_name)
        proxy = stream if isinstance(stream, _CaptureProxy) else _CaptureProxy(stream)
        proxy._active_captures += 1
        setattr(sys, stream_name, proxy)
    try:
        with proxy.capture() as buffer:
            yield buffer
    finally:
        with _capture_lock:
            # Captures running at the same time share one proxy, so only the
            # last one to finish puts the original stream back. The identity
            # check keeps us from doing so if something else has replaced
            # sys.<stream_name> in the meantime, as restoring would then throw
            # away their stream rather than ours.
            proxy._active_captures -= 1
            if proxy._active_captures == 0 and getattr(sys, stream_name) is proxy:
                setattr(sys, stream_name, proxy.wrapped_stream)
