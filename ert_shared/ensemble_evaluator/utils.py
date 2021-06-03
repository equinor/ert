import asyncio
import logging
import ssl
import time
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


def get_ssl_context(cert: Optional[str]) -> Optional[ssl.SSLContext]:
    if cert is None:
        return None
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.load_verify_locations(cadata=cert)
    return ssl_context


async def attempt_connection(
    url: str,
    token: Optional[str] = None,
    cert: Optional[str] = None,
    connection_timeout: float = 2,
) -> None:
    timeout = aiohttp.ClientTimeout(connect=connection_timeout)
    headers = {} if token is None else {"token": token}
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method="get",
            url=url,
            ssl=get_ssl_context(cert),
            headers=headers,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()


async def wait_for_evaluator(
    base_url: str,
    token: Optional[str] = None,
    cert: Optional[str] = None,
    healthcheck_endpoint: str = "/healthcheck",
    timeout: float = 60,
    connection_timeout: float = 2,
) -> None:
    healthcheck_url = base_url + healthcheck_endpoint
    start = time.time()
    sleep_time = 0.2
    sleep_time_max = 5.0
    while time.time() - start < timeout:
        try:
            await attempt_connection(
                url=healthcheck_url,
                token=token,
                cert=cert,
                connection_timeout=connection_timeout,
            )
            return
        except aiohttp.ClientError:
            sleep_time = min(sleep_time_max, sleep_time * 2)
            remaining_time = max(0, timeout - (time.time() - start) + 0.1)
            await asyncio.sleep(min(sleep_time, remaining_time))

    # We have timed out, but we make one last attempt to ensure that
    # we have tried to connect at both ends of the time window
    await attempt_connection(
        url=healthcheck_url,
        token=token,
        cert=cert,
        connection_timeout=connection_timeout,
    )
