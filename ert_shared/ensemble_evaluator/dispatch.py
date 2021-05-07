from collections import defaultdict

import asyncio
from typing import Optional


class Batcher:
    def __init__(self, timeout, loop=None):
        self._timeout = timeout
        self._running = True
        self.__LOOKUP_MAP = defaultdict(list)

        # Schedule task
        self._task = asyncio.ensure_future(self._job(), loop=loop)

    async def _work(self):
        for f in self.__LOOKUP_MAP:
            events, self.__LOOKUP_MAP[f] = self.__LOOKUP_MAP[f], []
            if events:
                await f(events)

    def put(self, f, event):
        self.__LOOKUP_MAP[f].append(event)

    async def _job(self):
        while self._running:
            await asyncio.sleep(self._timeout)
            await self._work()

        # Make sure no events are lingering
        await self._work()

    async def join(self):
        self._running = False
        await self._task


class Dispatcher:
    def __init__(self, batcher=None):
        self.__LOOKUP_MAP = defaultdict(list)
        self._batcher: Optional[Batcher] = batcher

    def register_event_handler(self, event_types, batching=False):
        def decorator(function):
            nonlocal event_types, batching
            if not isinstance(event_types, set):
                event_types = set({event_types})
            for event_type in event_types:
                self.__LOOKUP_MAP[event_type].append((function, batching))

            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)

            return wrapper

        return decorator

    async def handle_event(self, event):
        for f, batching in self.__LOOKUP_MAP[event["type"]]:
            if batching:
                if self._batcher is None:
                    raise RuntimeError(
                        f"Requested batching of {event} with handler {f}, but no batcher was specified"
                    )
                self._batcher.put(f, event)
            else:
                await f(event)
