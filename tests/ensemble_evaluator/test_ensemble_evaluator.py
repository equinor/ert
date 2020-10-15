from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
import websockets
import pytest


async def dispatcher():
    uri = f"ws://{ee._host}:{ee._port}/dispatch"
    async with websockets.connect(uri) as websocket:
        pass


@pytest.mark.skip(reason="to be fixed by @mbend")
def test_it():
    ee = EnsembleEvaluator()
    monitor = ee.run()

    i = 0
    for event in monitor.track():
        assert event._event_index == i
        i += 1
        if event.is_terminated():
            assert False
    ee.stop()
