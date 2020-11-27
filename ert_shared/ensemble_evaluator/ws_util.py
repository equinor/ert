import websockets
import asyncio
import logging

logger = logging.getLogger(__name__)


def wait_for_ws(url, max_retries=10):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(wait(url, max_retries))
    loop.close()


async def wait(url, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            async with websockets.connect(url):
                pass
            return
        except OSError as e:
            logger.info(f"{__name__} failed to connect ({retries}/{max_retries}: {e}")
            await asyncio.sleep(0.2 + 5 * retries)
            retries += 1
    raise ConnectionRefusedError(f"Could not connect to {url} after {retries} retries")
