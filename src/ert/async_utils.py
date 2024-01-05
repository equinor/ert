from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Generator,
    MutableSequence,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


@asynccontextmanager
async def background_tasks() -> AsyncGenerator[Any, Any]:
    """Context manager for long-living tasks that cancel when exiting the
    context

    """

    tasks: MutableSequence[asyncio.Task[Any]] = []

    def add(coro: Coroutine[Any, Any, Any]) -> None:
        tasks.append(asyncio.create_task(coro))

    try:
        yield add
    finally:
        for t in tasks:
            t.cancel()
        for exc in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(exc, asyncio.CancelledError):
                continue
            if isinstance(exc, BaseException):
                logger.error(str(exc), exc_info=exc)
        tasks.clear()


def new_event_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    loop.set_task_factory(_create_task)
    return loop


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(new_event_loop())
        return asyncio.get_event_loop()


def _create_task(
    loop: asyncio.AbstractEventLoop,
    coro: Union[Coroutine[Any, Any, _T], Generator[Any, None, _T]],
) -> asyncio.Task[_T]:
    task = asyncio.Task(coro, loop=loop)
    task.add_done_callback(_done_callback)
    return task


def _done_callback(task: asyncio.Task[_T_co]) -> None:
    assert task.done()
    try:
        if (exc := task.exception()) is None:
            return

        logger.error(f"Exception occurred during {task.get_name()}", exc_info=exc)
    except asyncio.CancelledError:
        pass
