import asyncio
import queue
import threading
from contextlib import contextmanager
from http import HTTPStatus

import aiohttp
import websockets

from ert.async_utils import get_event_loop
from ert.shared import port_handler

from .narrative import InteractionDirection, _Narrative


class NarrativeProxy:
    def __init__(self, narrative: _Narrative):
        self.narrative = narrative
        self.currentInteraction = narrative.interactions.pop()
        self.error = None

    async def _async_proxy(self, url, q):
        self.done = get_event_loop().create_future()

        async def handle_messages(msg_q: asyncio.Queue, done: asyncio.Future):
            try:
                for interaction in self.narrative.interactions:
                    await interaction.verify(msg_q.get)
            except Exception as e:
                done.set_result(e)

        async def handle_server(server, client, msg_q):
            async for msg in server:
                await msg_q.put((InteractionDirection.RESPONSE, msg))
                await client.send(msg)

        async def handle_client(client, _path):
            msg_q = asyncio.Queue()
            if _path == "/client":
                async with websockets.client.connect(url + _path) as server:
                    msg_task = asyncio.ensure_future(handle_messages(msg_q, self.done))
                    server_task = asyncio.ensure_future(
                        handle_server(server, client, msg_q)
                    )

                    async for msg in client:
                        await msg_q.put((InteractionDirection.REQUEST, msg))
                        await server.send(msg)
                    server_task.cancel()
                    await server_task
                    await msg_task

        async with websockets.server.serve(
            handle_client,
            host="localhost",
            port=0,
            process_request=self.process_request,
        ) as s:
            local_family = port_handler.get_family_for_localhost()
            port = [p.getsockname()[1] for p in s.sockets if p.family == local_family]
            if len(port) == 0:
                self.done.set_result(
                    aiohttp.ClientError("Unable to find suitable port for proxy")
                )

            get_event_loop().run_in_executor(None, lambda: q.put(port[0]))
            get_event_loop().run_in_executor(None, lambda: q.put(self.done))
            error = await self.done
            q.put(error)

    async def process_request(self, path, request_headers):
        if path == "/healthcheck":
            return HTTPStatus.OK, {}, b""

    def _proxy(self, url, q):
        q.put(get_event_loop())
        get_event_loop().run_until_complete(self._async_proxy(url, q))

    @contextmanager
    def proxy(self, url):
        q = queue.Queue()
        t = threading.Thread(target=self._proxy, args=(url, q))
        t.start()
        loop = q.get()
        port = q.get()
        done = q.get()
        yield port
        if not done.done():
            loop.call_soon_threadsafe(done.set_result, None)
        t.join()
        error = q.get()
        if error:
            raise error
