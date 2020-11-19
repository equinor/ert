import asyncio
import websockets

from ert_shared.ensemble_evaluator.config import load_config


from cloudevents.http.event import CloudEvent
from cloudevents.http import to_json


class _Ensemble:
    def __init__(self, reals, metadata):
        self._reals = reals
        self._metadata = metadata
        self._ee_config = load_config()

    def __repr__(self):
        return f"Ensemble with {len(self._reals)} members"

    def evaluate(self, host, port):
        pass

    def get_reals(self):
        return self._reals

    def get_metadata(self):
        return self._metadata

    def send_cloudevent(self, event):
        loop = asyncio.new_event_loop()

        async def _send(event):
            async with websockets.connect(
                self._ee_config.get("dispatch_url")
            ) as websocket:
                await websocket.send(to_json(event))

        loop.run_until_complete(_send(event))
        loop.close()
