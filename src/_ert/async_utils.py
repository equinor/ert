from __future__ import annotations

import asyncio
import logging
import traceback
from contextlib import suppress
from typing import Any, Coroutine, Generator, Mapping, TypeVar, Union

import uvloop

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


def new_event_loop() -> asyncio.AbstractEventLoop:
    loop = uvloop.new_event_loop()
    loop.set_task_factory(_create_task)
    return loop


def get_running_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def _create_task(
    loop: asyncio.AbstractEventLoop,
    coro: Union[Coroutine[Any, Any, _T], Generator[Any, None, _T]],
    **kwargs: Mapping[str, Any],
) -> asyncio.Task[_T]:
    assert asyncio.iscoroutine(coro)
    task = asyncio.Task(coro, loop=loop, **kwargs)  # type: ignore
    task.add_done_callback(_done_callback)
    return task


def _done_callback(task: asyncio.Task[_T_co]) -> None:
    assert task.done()
    with suppress(asyncio.CancelledError):
        if (exc := task.exception()) is None:
            return

        exc_traceback = "".join(
            traceback.format_exception(None, exc, exc.__traceback__)
        )
        logger.error(
            (
                f"Exception in scheduler task {task.get_name()}: {exc}\n"
                f"Traceback: {exc_traceback}"
            )
        )
        raise exc
