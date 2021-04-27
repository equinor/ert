import websockets
import asyncio
import logging
import ssl

from websockets.http import Headers

from ert_shared.ensemble_evaluator.client import Client

logger = logging.getLogger(__name__)


def wait_for_ws(url, token=None, cert=None, max_retries=10):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(wait(url, token, cert, max_retries))
    loop.close()


async def wait(url, token=None, cert=None, max_retries=10):
    retries = 0
    client = Client(url, token, cert)
    while retries < max_retries:
        try:
            ws = await client.get_websocket()
            await ws.close()
            return
        except OSError as e:
            logger.info(f"{__name__} failed to connect ({retries}/{max_retries}: {e}")
            await asyncio.sleep(0.2 + 5 * retries)
            retries += 1
    raise ConnectionRefusedError(f"Could not connect to {url} after {retries} retries")
