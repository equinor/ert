from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
import websockets
import asyncio

async def dispatcher():
    uri = f"ws://{ee._host}:{ee._port}/dispatch"
    async with websockets.connect(uri) as websocket:
        pass

def test_it():
    ee = EnsembleEvaluator()
    monitor = ee.run()


    loop = asyncio.get_event_loop()
    
    i = 0
    for event in monitor.track():
        assert event._event_index == i
        i += 1
        if event.is_terminated():
            assert False
    ee.stop()