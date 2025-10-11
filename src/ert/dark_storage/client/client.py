import ssl

import httpx
from httpx_retries import Retry, RetryTransport

from ._session import ConnInfo, find_conn_info


class Client(httpx.Client):
    """
    Wrapper class for httpx.Client that provides a user-friendly way to
    interact with ERT Storage's API

    Stores 'conn_info' to bridge the gap to the Everest client setup
    """

    def __init__(self, conn_info: ConnInfo | None = None) -> None:
        if conn_info is None:
            conn_info = find_conn_info()

        self.conn_info = conn_info

        headers = {}
        if conn_info.auth_token is not None:
            headers = {"Token": conn_info.auth_token}
        super().__init__(
            base_url=conn_info.base_url,
            headers=headers,
            transport=RetryTransport(
                httpx.HTTPTransport(
                    verify=ssl.create_default_context(cafile=conn_info.cert)
                    if isinstance(conn_info.cert, str)
                    else conn_info.cert,
                ),
                retry=Retry(total=5, backoff_factor=0.5),
            ),
            timeout=3,
        )
