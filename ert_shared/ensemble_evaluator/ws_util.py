import websockets
import asyncio
import logging


def wait_for_ws(url, max_retries=3):
    asyncio.get_event_loop().run_until_complete(wait(url, max_retries))


async def wait(url, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            async with websockets.connect(url):
                pass

            return
        except OSError as e:
            logging.info(f"{__name__} failed to connect ({retries}/{max_retries}: {e}")
            retries += 1
            await asyncio.sleep(0.2)
