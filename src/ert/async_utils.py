import asyncio
from typing import Awaitable, TypeVar


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        return asyncio.get_event_loop()


T = TypeVar("T")


def run_in_loop(coro: Awaitable[T]) -> T:
    return get_event_loop().run_until_complete(coro)
