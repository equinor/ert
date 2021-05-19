from collections import defaultdict, deque, OrderedDict

import asyncio
from typing import Optional


class Batcher:
    def __init__(self, timeout, loop=None):
        self._timeout = timeout
        self._running = True
        self._buffer = deque()

        # Schedule task
        self._task = asyncio.ensure_future(self._job(), loop=loop)

    async def _work(self):
        event_buffer, self._buffer = self._buffer, deque()
        if event_buffer:
            function_to_events_map = OrderedDict()
            for f, event in event_buffer:
                if f not in function_to_events_map:
                    function_to_events_map[f] = []
                function_to_events_map[f].append(event)
            for f, events in function_to_events_map.items():
                await f(events)

    def put(self, f, event):
        self._buffer.append((f, event))

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
