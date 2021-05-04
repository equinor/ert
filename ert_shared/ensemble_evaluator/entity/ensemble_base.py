from ert_shared.ensemble_evaluator.client import Client

from ert_shared.ensemble_evaluator.entity import serialization

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

    async def send_cloudevent(self, url, event, token=None, cert=None, retries=1):
        client = Client(url, token, cert)
        await client._send(
            to_json(event, data_marshaller=serialization.evaluator_marshaller)
        )
        await client.websocket.close()
