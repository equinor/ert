import httpx

from ._session import ConnInfo


class AsyncClient(httpx.AsyncClient):
    """
    Wrapper class for httpx.AsyncClient that provides a user-friendly way to
    interact with ERT Storage's API
    """

    def __init__(self, conn_info: ConnInfo) -> None:
        headers = {}
        if conn_info.auth_token is not None:
            headers = {"Token": conn_info.auth_token}

        super().__init__(base_url=conn_info.base_url, headers=headers)
