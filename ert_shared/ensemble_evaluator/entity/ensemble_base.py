import asyncio
import threading
import websockets

from ert_shared.ensemble_evaluator.config import load_config

from cloudevents.http import to_json


class _Ensemble:
    def __init__(self, reals, metadata):
        self._reals = reals
        self._metadata = metadata

    def __repr__(self):
        return f"Ensemble with {len(self._reals)} members"

    def evaluate(self, host, ee_id):
        pass

    def cancel(self):
        pass

    def is_cancellable(self):
        return False

    def get_reals(self):
        return self._reals

    def get_active_reals(self):
        return list(filter(lambda real: real.is_active(), self._reals))

    def get_metadata(self):
        return self._metadata

    async def send_cloudevent(self, url, event, retries=10):
        for retry in range(retries):
            try:
                async with websockets.connect(url) as websocket:
                    await websocket.send(to_json(event))
                return
            except ConnectionRefusedError:
                await asyncio.sleep(1)
        raise IOError(f"Could not send event {event}  to url {url}")
